"""Approved call rewards. Runtime parsing uses the pinned native vLLM parser."""
from collections import Counter
import json
import math
import re


def typed_key(value, numeric_equivalence=False):
    if numeric_equivalence and type(value) in (int, float):
        if not math.isfinite(value): raise ValueError('non-finite number')
        return ('number', value)
    if isinstance(value, dict):
        return ('dict', tuple(sorted((k, typed_key(v, numeric_equivalence)) for k,v in value.items())))
    if isinstance(value, list):
        return ('list', tuple(typed_key(v, numeric_equivalence) for v in value))
    if type(value) is float and not math.isfinite(value): raise ValueError('non-finite number')
    return (type(value).__name__, value)


def call_key(call, numeric_equivalence=False):
    return call['name'], typed_key(call['arguments'], numeric_equivalence)


def pair_score(pred, gold, numeric_equivalence=False):
    if pred['name'] != gold['name']: return 0.0
    a,b = pred['arguments'], gold['arguments']
    common = sum(k in b and typed_key(v,numeric_equivalence)==typed_key(b[k],numeric_equivalence) for k,v in a.items())
    f = 2*common/(len(a)+len(b)) if a or b else 1.0
    return .25 + .75*f


def maximum_assignment(weights):
    """Hungarian maximum-weight matching, padding with unmatched zero-weight calls."""
    n = max(len(weights), len(weights[0]) if weights else 0)
    if not n: return 0.0
    cost = [[-(weights[i][j] if i<len(weights) and j<len(weights[i]) else 0.) for j in range(n)] for i in range(n)]
    u,v,p,way = [0.]*(n+1),[0.]*(n+1),[0]*(n+1),[0]*(n+1)
    for i in range(1,n+1):
        p[0], j0 = i, 0
        minv,used = [math.inf]*(n+1),[False]*(n+1)
        while True:
            used[j0] = True; i0 = p[j0]; delta,j1 = math.inf,0
            for j in range(1,n+1):
                if not used[j]:
                    cur = cost[i0-1][j-1]-u[i0]-v[j]
                    if cur < minv[j]: minv[j],way[j] = cur,j0
                    if minv[j] < delta: delta,j1 = minv[j],j
            for j in range(n+1):
                if used[j]: u[p[j]] += delta; v[j] -= delta
                else: minv[j] -= delta
            j0 = j1
            if p[j0] == 0: break
        while True:
            j1 = way[j0]; p[j0] = p[j1]; j0 = j1
            if j0 == 0: break
    return sum(-cost[p[j]-1][j-1] for j in range(1,n+1))


def score_calls(pred, reference, mode, numeric_equivalence=False):
    gold = reference['calls']
    if reference['kind'] == 'no_call': return float(not pred)
    if not pred or not gold: return 0.0
    if mode == 'strict':
        return float(Counter(call_key(c,numeric_equivalence) for c in pred) == Counter(call_key(c,numeric_equivalence) for c in gold))
    if mode != 'partial': raise ValueError(mode)
    weights = [[pair_score(p,g,numeric_equivalence) for g in gold] for p in pred]
    return 2*maximum_assignment(weights)/(len(pred)+len(gold))


CALL = re.compile(r'<tool_call>\s*<function=([^<>\n]+)>(.*?)</function>\s*</tool_call>',re.S)
PARAM = re.compile(r'\s*<parameter=([^<>\n]+)>(.*?)</parameter>',re.S)
MARKUP = re.compile(r'</?(?:tool|function|parameter)',re.I)


def strict_xml(text):
    """Validate the complete tool markup before the permissive native parser runs.

    Allows normal prose outside calls. Rejects unmatched/nested tags, duplicate
    parameters, foreign call serializations, and calls only partly recovered.
    Returns raw (name, {parameter: XML text}) records, not typed arguments.
    """
    if not text.strip(): raise ValueError('empty')
    calls, end = [], 0
    for m in CALL.finditer(text):
        if MARKUP.search(text[end:m.start()]): raise ValueError('unparsed tool markup')
        body, args, pos = m[2], {}, 0
        for pm in PARAM.finditer(body):
            if body[pos:pm.start()].strip(): raise ValueError('text outside parameter')
            if pm[1] in args: raise ValueError('duplicate parameter')
            if MARKUP.search(pm[2]): raise ValueError('nested tool markup')
            value = pm[2]
            if value.startswith('\n'): value=value[1:]
            if value.endswith('\n'): value=value[:-1]
            args[pm[1]]=value; pos=pm.end()
        if body[pos:].strip(): raise ValueError('malformed parameter section')
        calls.append((m[1],args)); end=m.end()
    if MARKUP.search(text[end:]): raise ValueError('unparsed tool markup')
    if not calls and (re.match(r'^\s*\[\s*[^\n]+\(',text) or re.search(r'"(?:tool_calls|arguments)"\s*:',text)):
        raise ValueError('foreign call serialization')
    return calls


class NativeParser:
    def __init__(self, tokenizer):
        from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
        self.parser = Qwen3CoderToolParser(tokenizer)

    def parse(self,text,tools):
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        raw = strict_xml(text)
        request = ChatCompletionRequest.model_validate({'model':'Qwen/Qwen3.5-2B','messages':[{'role':'user','content':'parse'}],'tools':tools})
        self.parser.tools = request.tools
        parsed = self.parser.extract_tool_calls(text,request)
        calls = [{'name':c.function.name,'arguments':json.loads(c.function.arguments)} for c in parsed.tool_calls]
        if len(calls)!=len(raw): raise ValueError('native parser recovered only part of markup')
        for c,(name,params) in zip(calls,raw):
            if c['name']!=name or set(c['arguments'])!=set(params): raise ValueError('native parser disagreement')
        return calls


def score_text(text,tools,reference,parser,mode,truncated=False,numeric_equivalence=False):
    if truncated: return 0.0, {'valid':False,'error':'truncated','calls':[]}
    try:
        strict_xml(text)
        calls=parser.parse(text,tools)
        score=score_calls(calls,reference,mode,numeric_equivalence)
        return score, {'valid':True,'calls':calls}
    except (ValueError,TypeError,KeyError,OverflowError) as e:
        return 0.0, {'valid':False,'error':str(e),'calls':[]}


def make_reward(tokenizer,mode,max_completion_length,numeric_equivalence=False,model_eos_ids=None):
    parser=NativeParser(tokenizer)
    eos=tokenizer.eos_token_id
    eos_ids=set(eos if isinstance(eos,list) else [eos])
    if model_eos_ids is not None:
        eos_ids.update(model_eos_ids if isinstance(model_eos_ids,list) else [model_eos_ids])
    def reward(prompts,completions,completion_ids,reference_json,tools_json,**kwargs):
        scores=[]
        for text,ids,ref,tools in zip(completions,completion_ids,reference_json,tools_json,strict=True):
            # Decode again without discarding <tool_call>, which is a special token.
            truncated=not ids or ids[-1] not in eos_ids
            # EOS alone is an empty response, not a successful no-call answer.
            content_ids=ids[:-1] if ids and ids[-1] in eos_ids else ids
            text=tokenizer.decode(content_ids,skip_special_tokens=False)
            score,_=score_text(text,json.loads(tools),json.loads(ref),parser,mode,truncated,numeric_equivalence)
            scores.append(score)
        return scores
    reward.__name__=mode+'_call_reward'
    return reward
