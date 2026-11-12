"""Requires the real pinned vLLM package, but does not load model weights."""
import json
from pathlib import Path

from eda import MODEL_REVISION
from rewards import NativeParser,score_text


def main():
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('Qwen/Qwen3.5-2B',revision=MODEL_REVISION)
    parser=NativeParser(tokenizer)
    tools=[{'type':'function','function':{'name':"Agent's API / Search, V2",'parameters':{'type':'object','properties':{
        'from':{'type':'string'},'amount':{'type':'number'},'payload':{'type':'object'},'enabled':{'type':'boolean'}}}}}]
    calls=[{'name':tools[0]['function']['name'],'arguments':{'from':'x=y','amount':1.0,'payload':{'a':[1,True,'1']},'enabled':True}}]*2
    messages=[{'role':'user','content':'test'}]
    prefix=tokenizer.apply_chat_template(messages,tools=tools,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    full=tokenizer.apply_chat_template(messages+[{'role':'assistant','content':'','tool_calls':[{'type':'function','function':c}for c in calls]}],tools=tools,tokenize=False,enable_thinking=False)
    assert full.startswith(prefix)
    text=full[len(prefix):]
    score,info=score_text(text,tools,{'kind':'calls','calls':calls},parser,'strict',numeric_equivalence=True)
    assert score==1,(text,info)
    assert score_text(text,tools,{'kind':'no_call','calls':[]},parser,'partial',numeric_equivalence=True)[0]==0
    for broken in (text.replace('</parameter>','',1),text.replace('</tool_call>','',1),'',text):
        s,detail=score_text(broken,tools,{'kind':'calls','calls':calls},parser,'strict',truncated=broken==text,numeric_equivalence=True)
        assert s==0 and not detail['valid']
    print(json.dumps({'native_vllm_parser':'passed','qwen_template':'passed','repeated_calls':'passed','numeric_equivalence':'passed','raw_target':text},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
