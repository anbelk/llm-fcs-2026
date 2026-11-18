import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Annotated, Literal

import torch
import typer
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForImageTextToText, AutoTokenizer, TrainerCallback
from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available
from trl import GRPOConfig, GRPOTrainer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

from rewards import score_calls

MODEL = 'Qwen/Qwen3.5-2B'
REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'
CATEGORIES = ('simple', 'multiple', 'parallel', 'parallel_multiple', 'no_call')


def reward_function(tokenizer, reward_index, completion_limit, legacy=False):
    parser = Qwen3CoderToolParser(tokenizer)

    def reward(completions, completion_ids, tools_json, expected_json, **_):
        scores = []
        for text, ids, schemas, reference in zip(
            completions, completion_ids, tools_json, expected_json, strict=True
        ):
            request = ChatCompletionRequest.model_validate({
                'model': MODEL, 'messages': [{'role': 'user', 'content': ''}],
                'tools': json.loads(schemas),
            })
            parser.tools = request.tools
            parsed = parser.extract_tool_calls(text, request)
            try:
                calls = [{'name': call.function.name,
                          'arguments': json.loads(call.function.arguments)}
                         for call in parsed.tool_calls]
            except (TypeError, ValueError):
                scores.append(0.0)
                continue
            valid = (
                len(ids) < completion_limit or
                (bool(ids) and ids[-1] == tokenizer.eos_token_id)
            ) and text.count('<tool_call>') == len(calls) and all(
                text.count(start) == text.count(end) for start, end in
                (('<tool_call>', '</tool_call>'), ('<function=', '</function>'),
                 ('<parameter=', '</parameter>'))
            )
            scores.append(score_calls(json.loads(reference), calls, text, valid, normalize_numbers=not legacy)[reward_index])
        return scores

    reward.__name__ = ('strict_reward', 'partial_reward')[reward_index]
    return reward


def load_split(path, tokenizer, max_prompt_tokens, sample_size, seed, stratified=True):
    rows = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        prompt = tokenizer.apply_chat_template(
            record['prompt'], tools=record['tools'], add_generation_prompt=True,
            tokenize=False, enable_thinking=False,
        )
        if len(tokenizer(prompt, add_special_tokens=False)['input_ids']) > max_prompt_tokens:
            continue
        row = {'prompt': prompt, 'tools_json': json.dumps(record['tools']),
               'expected_json': json.dumps(record['expected_calls']), 'category': record['category']}
        if stratified:
            row.update(dialogue_id=record['dialogue_id'], turn=record['turn'])
        rows.append(row)
    shuffled = Dataset.from_list(rows).shuffle(seed=seed)
    if not stratified:
        if sample_size and len(shuffled) > sample_size:
            shuffled = shuffled.select(range(sample_size))
        if not len(shuffled):
            raise ValueError(f'No examples remain in {path}')
        print(path.name, 'examples after filtering and selection:', len(shuffled), flush=True)
        return shuffled
    buckets = {category: [] for category in CATEGORIES}
    for row in shuffled:
        buckets[row['category']].append(row)
    if any(not bucket for bucket in buckets.values()):
        raise ValueError(f'Missing category in {path}')
    selected = []
    for index in range(max(map(len, buckets.values()))):
        for category in CATEGORIES:
            if index < len(buckets[category]):
                selected.append(buckets[category][index])
                if sample_size and len(selected) == sample_size:
                    break
        if sample_size and len(selected) == sample_size:
            break
    print(path.name, 'selected:', len(selected), dict(Counter(r['category'] for r in selected)), flush=True)
    return Dataset.from_list(selected)


class ValidationCallback(TrainerCallback):
    def __init__(self, tokenizer, validation, output, interval, patience, completion_limit, evaluation_steps=(), checkpoint_on_eval=False):
        self.tokenizer = tokenizer
        self.validation = validation
        self.output = output
        self.interval = interval
        self.patience = patience
        self.completion_limit = completion_limit
        self.reward = reward_function(tokenizer, 0, completion_limit)
        self.best_score = -1.0
        self.best_step = 0
        self.no_improvement = 0
        self.evaluation_steps = set(evaluation_steps)
        self.checkpoint_on_eval = checkpoint_on_eval

    def check(self, model, step, control):
        was_training = model.training
        model.eval()
        grouped = defaultdict(list)
        details = []
        with torch.inference_mode():
            for start in range(0, len(self.validation), 4):
                batch = self.validation.select(range(start, min(start + 4, len(self.validation))))
                encoded = self.tokenizer(list(batch['prompt']), return_tensors='pt', padding=True,
                                         add_special_tokens=False).to(model.device)
                output = model.generate(**encoded, max_new_tokens=self.completion_limit,
                                        do_sample=False, use_cache=True,
                                        pad_token_id=self.tokenizer.pad_token_id,
                                        eos_token_id=self.tokenizer.eos_token_id)
                generated = output[:, encoded['input_ids'].shape[1]:].tolist()
                ids = []
                for sequence in generated:
                    if self.tokenizer.eos_token_id in sequence:
                        sequence = sequence[:sequence.index(self.tokenizer.eos_token_id) + 1]
                    ids.append(sequence)
                texts = self.tokenizer.batch_decode(ids, skip_special_tokens=True)
                scores = self.reward(texts, ids, batch['tools_json'], batch['expected_json'])
                for row, text, score in zip(batch, texts, scores, strict=True):
                    grouped[row['category']].append(score)
                    details.append({'step': step, 'dialogue_id': row['dialogue_id'],
                                    'turn': row['turn'], 'category': row['category'],
                                    'score': score, 'response': text})
        if was_training:
            model.train()
        count = sum(len(v) for v in grouped.values())
        result = {'step': step, 'strict_accuracy': sum(map(sum, grouped.values())) / count,
                  'categories': {k: sum(v) / len(v) for k, v in grouped.items()}, 'count': count}
        with (self.output / 'validation.jsonl').open('a') as f:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
        with (self.output / 'validation_predictions.jsonl').open('a') as f:
            for item in details:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        if result['strict_accuracy'] > self.best_score:
            self.best_score = result['strict_accuracy']
            self.best_step = step
            self.no_improvement = 0
            model.save_pretrained(self.output / 'best-adapter')
        else:
            self.no_improvement += 1
            if self.no_improvement >= self.patience:
                control.should_training_stop = True
        print('validation:', json.dumps(result), 'best_step:', self.best_step,
              'checks_without_improvement:', self.no_improvement, flush=True)
        return control

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        return self.check(model, 0, control)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        selected = (state.global_step in self.evaluation_steps if self.evaluation_steps else
                    state.global_step and state.global_step % self.interval == 0)
        if selected:
            control = self.check(model, state.global_step, control)
            if self.checkpoint_on_eval:
                control.should_save = True
        return control



def main(
    run: Annotated[Literal['grpo-strict', 'cispo-strict', 'cispo-partial', 'grpo-partial-long'], typer.Argument()],
    output: Annotated[Path, typer.Option()],
    data: Path = Path('data'),
):
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A CUDA GPU with bfloat16 support is required')
    if output.exists():
        raise ValueError(f'Output already exists: {output}')
    long = run == 'grpo-partial-long'
    if long != bool(is_fast_path_available):
        if long:
            raise RuntimeError('The long run requires the causal-conv1d fast path; see README')
        raise RuntimeError('The archived runs require the original fallback kernel; see README')
    output.mkdir(parents=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tokenizer.padding_side = 'left'
    if long:
        train = load_split(data / 'train.jsonl', tokenizer, 2048, 500, 42)
        validation = load_split(data / 'validation.jsonl', tokenizer, 2048, 150, 42)
        for name, dataset in [('train_selection', train), ('validation_selection', validation)]:
            with (output / f'{name}.jsonl').open('w') as f:
                for row in dataset:
                    f.write(json.dumps({k: row[k] for k in ('dialogue_id', 'turn', 'category')}) + '\n')
    else:
        train = load_split(data / 'train.jsonl', tokenizer, 2048, 0, 42, stratified=False)
        validation = load_split(data / 'validation.jsonl', tokenizer, 2048, 8, 42, stratified=False)
    algorithm = 'cispo' if run.startswith('cispo') else 'grpo'
    reward_index = int(run in ('cispo-partial', 'grpo-partial-long'))
    completion_limit = 512 if long else 256
    options = GRPOConfig(
        output_dir=str(output), seed=42, data_seed=42,
        max_steps=2000 if long else 100, logging_steps=1,
        eval_strategy='no' if long else 'steps',
        save_strategy='no' if long else 'steps',
        **({} if long else {'eval_steps': 100, 'save_steps': 100, 'per_device_eval_batch_size': 4}),
        per_device_train_batch_size=4 if long else 1,
        gradient_accumulation_steps=1 if long else 4,
        num_generations=4, num_iterations=2, max_completion_length=completion_limit,
        temperature=0.8, learning_rate=1e-5 if long else 5e-6, bf16=True,
        loss_type=algorithm, scale_rewards='group',
        epsilon_high=4.0 if algorithm == 'cispo' else 0.2,
        mask_truncated_completions=True, beta=0.0,
        remove_unused_columns=False, log_completions=True, report_to='none',
    )
    model = AutoModelForImageTextToText.from_pretrained(MODEL, revision=REVISION, dtype=torch.bfloat16)
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                      target_modules=r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj)$')
    callback = ValidationCallback(tokenizer, validation, output, 200, 10, completion_limit,
                                  evaluation_steps=(120, 400, 1000, 2000), checkpoint_on_eval=True) if long else None
    trainer = GRPOTrainer(
        model=model, args=options,
        reward_funcs=[reward_function(tokenizer, reward_index, completion_limit, legacy=not long)],
        train_dataset=train, eval_dataset=None if long else validation,
        processing_class=tokenizer, peft_config=lora, callbacks=[callback] if callback else None,
    )
    (output / 'run_config.json').write_text(json.dumps({
        'run': run, 'data': str(data), 'reward': 'partial' if reward_index else 'strict',
        'legacy_reward': not long, 'max_prompt_tokens': 2048,
        'train_sample': 500 if long else 0, 'validation_sample': 150 if long else 8,
        'evaluation_steps': [120, 400, 1000, 2000] if long else [100],
        'trl': options.to_dict(), 'base_revision': REVISION,
    }, indent=2, default=str) + '\n')
    trainer.train()
    trainer.save_state()
    trained = trainer.accelerator.unwrap_model(trainer.model)
    if not isinstance(trained, PeftModel):
        raise RuntimeError('Expected a PEFT checkpoint to merge')
    if long:
        trained.save_pretrained(output / 'final-adapter')
        trained.load_adapter(str(output / 'best-adapter'), adapter_name='best')
        trained.set_adapter('best')
        (output / 'selection.json').write_text(json.dumps({
            'best_step': callback.best_step, 'best_strict_accuracy': callback.best_score,
            'final_step': trainer.state.global_step,
        }, indent=2) + '\n')
    else:
        trainer.save_model(str(output / 'adapter'))
    trained.merge_and_unload().save_pretrained(output / 'merged')
    tokenizer.save_pretrained(output / 'merged')


if __name__ == '__main__':
    typer.run(main)
