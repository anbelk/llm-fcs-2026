import itertools
import json
import unittest
from unittest.mock import patch

from rewards import maximum_assignment, pair_score, score_calls, score_text, strict_xml


def call(name='f', **args): return {'name':name,'arguments':args}
def ref(*calls): return {'kind':'calls' if calls else 'no_call','calls':list(calls)}


class Rewards(unittest.TestCase):
    def test_strict_multiset_and_types(self):
        gold=ref(call(x=1),call(x=2),call(x=1))
        self.assertEqual(score_calls([call(x=1),call(x=1),call(x=2)],gold,'strict'),1)
        self.assertEqual(score_calls([call(x=1),call(x=2)],gold,'strict'),0)
        for value in (True,1.,'1'):
            self.assertEqual(score_calls([call(x=value)],ref(call(x=1)),'strict'),0)
        self.assertEqual(score_calls([call(x=1.)],ref(call(x=1)),'strict',True),1)
        self.assertEqual(score_calls([call(x=True)],ref(call(x=1)),'strict',True),0)

    def test_nested_and_optional(self):
        gold=ref(call(x={'a':[1,{'b':True}], 'c':'2'}))
        self.assertEqual(score_calls([call(x={'c':'2','a':[1,{'b':True}]})],gold,'strict'),1)
        self.assertEqual(score_calls([call(x={'a':[1,{'b':1}],'c':'2'})],gold,'strict'),0)
        self.assertEqual(score_calls([call(x=1,optional=5)],ref(call(x=1)),'strict'),0)
        self.assertAlmostEqual(pair_score(call(x={'a':2}),call(x={'a':1})),.25)

    def test_partial_missing_extra_and_repeated(self):
        self.assertEqual(pair_score(call(),call()),1)
        self.assertEqual(pair_score(call('g',x=1),call(x=1)),0)
        self.assertAlmostEqual(pair_score(call(x=1),call(x=1,y=2)),.75)
        self.assertAlmostEqual(pair_score(call(x=1,y=2),call(x=1)),.75)
        self.assertAlmostEqual(score_calls([call(x=1)],ref(call(x=1),call(x=2)),'partial'),2/3)
        self.assertAlmostEqual(score_calls([call(x=1),call(x=1)],ref(call(x=1)),'partial'),2/3)
        self.assertEqual(score_calls([],ref(call()),'partial'),0)

    def test_matching_is_optimal(self):
        weights=[[.9,.8],[.85,.1]]  # greedy .9+.1 is wrong
        self.assertAlmostEqual(maximum_assignment(weights),1.65)
        for w in ([[1,.4,0],[.3,.8,.9],[.6,0,.2]], [[1],[.5]], [[.3,.8,.6]]):
            n=max(len(w),len(w[0])); padded=[[w[i][j] if i<len(w) and j<len(w[i]) else 0 for j in range(n)]for i in range(n)]
            brute=max(sum(padded[i][p[i]] for i in range(n))for p in itertools.permutations(range(n)))
            self.assertAlmostEqual(maximum_assignment(w),brute)

    def test_format_and_no_call(self):
        class Parser:
            def parse(self,text,tools): return [call(n) for n,_ in strict_xml(text)]
        parser=Parser()
        good='<tool_call>\n<function=Some name / API>\n<parameter=from>\nx\n</parameter>\n</function>\n</tool_call>'
        self.assertEqual(strict_xml(good),[('Some name / API',{'from':'x'})])
        for text in ('', ' ', '<tool_call>', '</tool_call>', '<tool_call><function=f></function>', '<function=f></function>',
                     '<tool_call><function=f><parameter=x>1</function></tool_call>',
                     '<tool_call><function=f><parameter=x>1</parameter><parameter=x>2</parameter></function></tool_call>',
                     '[f(x=1)]', '{"tool_calls":[]}'):
            score,info=score_text(text,[],ref(),parser,'strict')
            self.assertEqual(score,0,text); self.assertFalse(info['valid'],text)
        self.assertEqual(score_text('Which city?',[],ref(),parser,'strict')[0],1)
        self.assertEqual(score_text(good,[],ref(),parser,'partial')[0],0)
        self.assertEqual(score_text('Which city?',[],ref(),parser,'strict',truncated=True)[0],0)

    def test_eos_only_is_empty_and_model_eos_is_supported(self):
        from rewards import make_reward
        class Tokenizer:
            eos_token_id=9
            def decode(self,ids,**kwargs):return 'Which city?' if ids else ''
        class Parser:
            def parse(self,text,tools):return []
        with patch('rewards.NativeParser',return_value=Parser()):
            reward=make_reward(Tokenizer(),'strict',1024,True,[9,10])
            kwargs={'prompts':['x'],'completions':['unused'],'reference_json':[json.dumps(ref())],'tools_json':['[]']}
            self.assertEqual(reward(completion_ids=[[9]],**kwargs),[0])
            self.assertEqual(reward(completion_ids=[[2,9]],**kwargs),[1])
            self.assertEqual(reward(completion_ids=[[2,10]],**kwargs),[1])
            self.assertEqual(reward(completion_ids=[[2]],**kwargs),[0])


if __name__=='__main__': unittest.main()
