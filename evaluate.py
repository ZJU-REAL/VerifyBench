import jsonlines
import vllm
import re
import json
from collections import defaultdict


def load_dataset(args):
    data_file = 'verify_bench_hard' if args.hard else 'verify_bench'
    prompt_file = 'wo_ref' if args.wo_ref else 'w_ref'
    
    with open(f'prompt/{prompt_file}.txt', 'r') as f:
        prompt_template = f.read()
    
    data = []
    with jsonlines.open(f"data/{data_file}.jsonl", 'r') as reader:
        for item in reader:
            item['prompt'] = prompt_template.\
                replace('{question}', item['question']).\
                replace('{answer}', item['answer']).\
                replace('{completion}', item['completion'])
            data.append(item)
    return data


def inference(data, args):
    model = vllm.LLM(args.model_name_or_path)
    sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=2048)
    
    messages = []
    for item in data:
        messages.append([{"role": "user", "content": item['prompt']}])
    
    outputs = model.chat(messages, sampling_params)
    return outputs
    

def evaluate(data, outputs):
    
    def answer_parse(output):
        try:
            if re.findall(r'Yes|No', output)[-1] == "Yes":
                return True
            else:
                return False
        except Exception as e:
            return None
    
    case_by_case_results = []
    total_correct = 0
    total_count = 0
    
    for item, output in zip(data, outputs):
        item['model_generated_text'] = output.outputs[0].text
        item['judge'] = answer_parse(output.outputs[0].text)
        item['model_correct'] = (item['judge'] == item['gold_correct'])
        
        if item['model_correct']:
           total_correct += 1
           
        total_count += 1
        case_by_case_results.append(item)
    
    result_by_type = defaultdict(list)
    result_by_subtype = defaultdict(list)
    
    for item in case_by_case_results:
        result_by_type[item['answer_type']].append(item['model_correct'])
        result_by_subtype[item['answer_subtype']].append(item['model_correct'])
    
    acc_by_type = {k: sum(v) / len(v) for k, v in result_by_type.items()}
    acc_by_subtype = {k: sum(v) / len(v) for k, v in result_by_subtype.items()}

    return {
        "total_correct": total_correct,
        "total_count": total_count,
        "accuracy": total_correct / total_count,
        "accuracy_by_type": acc_by_type,
        "accuracy_by_subtype": acc_by_subtype,
        "case_by_case_results": case_by_case_results
    }    


def info(result):
    print('-'*10 + ' Evaluation Report ' + '-'*10)
    print(f"{'Average':30s} {result['accuracy'] * 100:>8.2f}")
    for k, v in sorted(result['accuracy_by_type'].items()):
       print(f"{k:30s} {v * 100:>8.2f}")
    for k, v in sorted(result['accuracy_by_subtype'].items()):
       print(f"{k:30s} {v * 100:>8.2f}")
    print('-'*10 + len(' Evaluation Report ') * '-' + '-'*10)   


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', '-m', type=str)
    parser.add_argument('--hard', action='store_true', )
    parser.add_argument('--wo-ref', action='store_true', )
    parser.add_argument('--output-file', '-o', type=str, default=None) # json
    args = parser.parse_args()

    data = load_dataset(args)
    outputs = inference(data, args)
    results = evaluate(data, outputs)
    info(results)
    
    if args.output_file is not None:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
            print(f"Results saved to {args.output_file}")
