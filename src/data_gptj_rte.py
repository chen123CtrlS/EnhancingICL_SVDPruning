import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from my_utils.data_load import load_dataset_train_and_test
from my_utils.data_process_cl import sample_data_rte,cat_demo_rte
from my_utils.metrics import normalize_answer,calculate_metric
from transformers import AutoTokenizer
from transformers import GPTJForCausalLM
import torch
from tqdm import tqdm
from laser.log_utils import Logger
from laser.LaserWrapper import LaserWrapper
import warnings
warnings.filterwarnings('ignore')


search = True
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
few_shot = True

# TODO hyperparametric
save_dir = f"/home/EnhancingICL_SVDPruning/src/results/"
logger = Logger(save_dir=save_dir, fname=f"test.txt")
# choices=['k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_up', 'fc_out', 'None', 'dont']
l_name = "mlp"
# list(range(-1, 27))
task_name  = 'rte'
local_data_path = r'' #TODO:your local dataset (if yes, otherwise '')
l_num = 26
rates = [0,1.0,5.0,7.5,9.0,9.5,9.9,9.95]
demo_size = 5
eval_data_size = 200
llm_name = "GPTJ"
llm_path = "EleutherAI/gpt-j-6B"
txtname = llm_name+'_'+task_name+'_'+l_name+str(l_num)+'.txt'
save_path = r'/home/EnhancingICL_SVDPruning/src/results/gptj/'

# load data
dataset = load_dataset_train_and_test(task_name=task_name,local_data_path=local_data_path)
# process data
demo_sample,eval_sample,_ = sample_data_rte(dataset['train'],seed=66, actual_sample_size = eval_data_size, demo_size = demo_size)
eval_data = cat_demo_rte(demo_sample,eval_sample,task_name, demo_shot=demo_size)
test_data = cat_demo_rte(demo_sample,dataset['test'],task_name, demo_shot=demo_size)
print('finish loading and processing ## {} ## data!'.format(task_name))

if search == False:
    run_test_number = min(len(test_data),500)
    eval_data=eval_data[:run_test_number]
    test_data=test_data[:run_test_number]
# load model
tokenizer = AutoTokenizer.from_pretrained(llm_path)
model = GPTJForCausalLM.from_pretrained(
        llm_path,
        revision="float16",
        torch_dtype=torch.float16
    )
# Space before true is important otherwise we will get the wrong token_id
true_token_ids = tokenizer(" Yes")
print(true_token_ids)
assert len(true_token_ids["input_ids"]) == 1
true_token_id = int(true_token_ids["input_ids"][0])

# Space before false is important otherwise we will get the wrong token_id
false_token_ids = tokenizer(" No")
print(false_token_ids)
assert len(false_token_ids["input_ids"]) == 1
false_token_id = int(false_token_ids["input_ids"][0])

def test(model,test_data):
    model = model.to(device)
    acc = .0
    for i,item in enumerate(tqdm(test_data)):
        premise = item['premise']
        hypothesis = item['hypothesis']
        label = item['label']
        if few_shot:
            demo = item['demo_sentence']
        else:
            demo = ''
        # <premise> Does this mean that <hypothesis> is true? Yes 0 or No 1?
        prompt = f"{premise} Does this mean that {hypothesis} is true? select Yes or No? "
        inputs = tokenizer(demo+prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            results = model(inputs.input_ids)
            logits = results.logits[0]                                      # question length x vocab
            log_prob = torch.nn.functional.log_softmax(logits, dim=1)       # question length x vocab

            last_token_logprob = log_prob[-1]                               # vocab

            true_logprob = last_token_logprob[true_token_id].item()
            false_logprob = last_token_logprob[false_token_id].item()

            if label == 0:     # Answer is Yes
                answer_log_prob = true_logprob
                is_correct = true_logprob > false_logprob
                answer = "Yes"
            else:               # Answer is No
                answer_log_prob = false_logprob
                is_correct = true_logprob < false_logprob
                answer = "No"
            if is_correct:
                acc+=1
            sorted_logprob, sorted_indices = torch.sort(last_token_logprob, descending=True)
            top_k_logprob = sorted_logprob[:10].detach().cpu().numpy()
            top_k_indices = sorted_indices[:10].detach()

            decoded_tokens = tokenizer.batch_decode(top_k_indices)
            top_k_tokens = [token for token in decoded_tokens]
            print(top_k_tokens)
            print('ans:{}'.format(answer))
    acc = acc/i
    return acc

if search:
    f1_s = []
    with open(save_path+txtname, 'w', encoding='utf-8') as f:
        for rate in rates:
            print(l_name+' '+str(l_num)+' '+str(rate)+' ')
            f.write(l_name+' '+str(l_num)+' '+str(rate)+' ')
            model_edit = LaserWrapper.get_edited_model(model=model,
                                                                    lname=l_name,
                                                                    lnum= l_num,
                                                                    rate=rate,
                                                                    intervention="rank-reduction",
                                                                    logger=logger,
                                                                    in_place=True)
            f1 = test(model_edit,eval_data)
            print(f1)
            f1_s.append(f1)
            f.write(str(f1)+'\n')

        best_clip_rate_id = torch.argmax(torch.tensor(f1_s)).item()
        best_clip_rate = rates[best_clip_rate_id]
        f.write("best_clip_rate: {}\n".format(best_clip_rate))
    del model_edit
else:
    best_clip_rate = 9.9
print('best_clip_rate'+': '+str(best_clip_rate))
# clear cache
del model 
torch.cuda.empty_cache()
# reload weight
model = GPTJForCausalLM.from_pretrained(
        llm_path,
        revision="float16",
        torch_dtype=torch.float16
    )
f1_test = test(model,test_data)
model_edit = LaserWrapper.get_edited_model(model=model,
                                                                lname = l_name,
                                                                lnum = l_num,
                                                                rate = best_clip_rate,
                                                                intervention="rank-reduction",
                                                                logger=logger,
                                                                in_place=True)
f1_test_svd= test(model_edit,test_data) 
print('test acc:{} -> {}'.format(f1_test,f1_test_svd))
if search:
    with open(save_path+txtname, 'a+', encoding='utf-8') as f:
        f.write(('test acc:{} -> {}'.format(f1_test,f1_test_svd)))