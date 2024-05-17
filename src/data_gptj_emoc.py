import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from my_utils.data_load import load_dataset_train_and_test
from my_utils.data_process_cl import sample_data_emo,cat_demo_emo
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
l_name = "mlp"
# list(range(-1, 27))
task_name  = 'emoc'
local_data_path = r'' #TODO:your local dataset (if yes, otherwise '')
l_num = 26
rates = [0,1.0,5.0,7.5,9.0,9.5,9.9,9.95]
llm_name = "GPTJ"
llm_path = "EleutherAI/gpt-j-6B"
txtname = llm_name+'_'+task_name+'_'+l_name+str(l_num)+'.txt'
save_path = r'/home/EnhancingICL_SVDPruning/src/results/gptj/'

# load data
dataset = load_dataset_train_and_test(task_name=task_name,local_data_path=local_data_path)
demo_per_shot = 5 # per class shot number
demo_sample,val_sample,_ = sample_data_emo(dataset = dataset['train'],actual_sample_size = 2500, demo_size = demo_per_shot,seed = 72)
test_sample = dataset['test']
eval_data = cat_demo_emo(demo_sample,val_sample,task_name,demo_per_shot*2)
test_data = cat_demo_emo(demo_sample,test_sample,task_name,demo_per_shot*2)
print('finish loading and processing ## {} ## data!'.format(task_name))
if search == False:
    run_test_number = min(len(test_data),50)
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
true_token_ids = tokenizer(" pos")
print(true_token_ids)
assert len(true_token_ids["input_ids"]) == 1
true_token_id = int(true_token_ids["input_ids"][0])

# Space before false is important otherwise we will get the wrong token_id
false_token_ids = tokenizer(" neg")
print(false_token_ids)
assert len(false_token_ids["input_ids"]) == 1
false_token_id = int(false_token_ids["input_ids"][0])


def test(model,test_data):
    model = model.to(device)
    i = 0
    acc =.0
    for item in tqdm(test_data):
        question = item["text"]
        label = item["label"]
        if label == 0:
            answer_ix = 1
        else:
            answer_ix = 0
        if few_shot:
            demo_shot = item['demo_sentence']
            prompted_question = 'text: ' + question.strip() +'\n sentiment :'
        else:
            demo_shot= ''
            prompted_question = "Consider the following text: " + \
                                                        question.strip() + ". Is this text sentiment pos or neg. The sentiment is"
                        
        inputs = tokenizer(demo_shot+ prompted_question, return_tensors="pt").to(device)


        with torch.no_grad():
                    # Compute log probability of question
            results = model(inputs.input_ids)
            logits = results.logits[0]                                      # question length x vocab
            log_prob = torch.nn.functional.log_softmax(logits, dim=1)       # question length x vocab

            last_token_logprob = log_prob[-1]                               # vocab

            true_logprob = last_token_logprob[true_token_id].item()
            false_logprob = last_token_logprob[false_token_id].item()

            if answer_ix == 1:     # Answer is True
                answer_log_prob = true_logprob
                is_correct = true_logprob > false_logprob
                answer = "pos"
            else:               # Answer is False
                answer_log_prob = false_logprob
                is_correct = true_logprob < false_logprob
                answer = "neg"
            if is_correct:
                acc+=1
            sorted_logprob, sorted_indices = torch.sort(last_token_logprob, descending=True)
            top_k_logprob = sorted_logprob[:10].detach().cpu().numpy()
            top_k_indices = sorted_indices[:10].detach()

            decoded_tokens = tokenizer.batch_decode(top_k_indices)
            top_k_tokens = [token for token in decoded_tokens]
            # print(top_k_tokens)
            # print('ans:{}'.format(answer))
            assert len(top_k_tokens) == 10
            i = i+1
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
    best_clip_rate = 1
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