import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from my_utils.data_load import load_dataset_train_and_test
from my_utils.data_process_qa import sample_data_squad,cat_demo_squad
from my_utils.metrics import normalize_answer,calculate_metric
from transformers import LlamaForCausalLM,LlamaTokenizerFast
import torch
from tqdm import tqdm
from laser.log_utils import Logger
from laser.LaserWrapper import LaserWrapper
import warnings
warnings.filterwarnings('ignore')
save_path = r'EnhancingICL_SVDPruning/src/results/llama2/'
local_data_path = r'' #TODO:your local dataset (if yes, otherwise '')
llm_path = r'Llama2-7B-chat' #TODO: your local model path
save_dir = r"EnhancingICL_SVDPruning/src/results/"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
few_shot = True
search = True #True for search clip rate, False for test on selected clip rate
def test(model,test_data):
    model = model.to(device)
    pairs = []
    n = len(test_data)
    for i,item in enumerate(tqdm(test_data)):
        title = item['title']
        context = item['context']
        question = item['question']
        answers = item['answers']
        if few_shot:
            demo_shot = item['demo_sentence']
        else:
            demo_shot= ''
        prompted_question = f"title: {title}\ncontext: {context}\nquestion: {question}\nanswer:"
        inputs = tokenizer(demo_shot+ prompted_question, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'], 
                max_length=len(inputs['input_ids'][0])+50,       
                num_return_sequences=1, 
                do_sample=False ,       # Turn off sampling and use greedy decoding
            )
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            pred = generated_text.split('answer')[-1].split('\n')[0][1:]
        except:
            pred = generated_text.split('answer')[-1]
        # print((pred,answers))
        pairs.append((pred,answers))
        # example:(' 2015 season', ['2015', 'the 2015 season', '2015'])
        if (i+1)%1000==0:
            print('data number {}/{}  f1 {}'.format(i+1,calculate_metric(pairs, n, 'f1')))
    f1 = calculate_metric(pairs, 'f1')
    return f1,pairs



# load data
task_name = 'squad'
dataset = load_dataset_train_and_test(task_name=task_name,local_data_path=local_data_path)
# process data
demo_size = 5
eval_data_size = 2000
demo_sample,eval_sample,_ = sample_data_squad(dataset['train'],seed=66, actual_sample_size = eval_data_size, demo_size = 10)
eval_data = cat_demo_squad(demo_sample,eval_sample,task_name, demo_shot=10)
test_data = cat_demo_squad(demo_sample,dataset['test'],task_name, demo_shot=10)
print('finish loading and processing ## {} ## data!'.format(task_name))
if search == False:
    run_test_number = min(len(test_data),50)
    eval_data=eval_data[:run_test_number]
    test_data=test_data[:run_test_number]
# load model
llm_name = "Llama2-7G"
tokenizer = LlamaTokenizerFast.from_pretrained(llm_path)
model = LlamaForCausalLM.from_pretrained(llm_path, revision="float16",torch_dtype=torch.float16).to(device)

# hyperparametric
logger = Logger(save_dir=save_dir, fname=f"test.txt")
# ['k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_up', 'fc_out', 'None', 'dont', 'all', 'mlp', 'attn']
l_name = "mlp"
# list(range(-1, 31))
l_num = 30
rates = [
    0,
    1.0,5.0,
         7.5,
         9.0,9.5,
         9.9,
         9.95]

txtname = llm_name+'_'+task_name+'_'+l_name+str(l_num)+'.txt'
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
            f1,pairs = test(model_edit,eval_data)
            print(f1)
            f1_s.append(f1)
            f.write(str(f1)+'\n')

        best_clip_rate_id = torch.argmax(torch.tensor(f1_s)).item()
        best_clip_rate = rates[best_clip_rate_id]
        f.write("best_clip_rate: {}\n".format(best_clip_rate))
    del model_edit
else:
    best_clip_rate = 1.0
print('best_clip_rate'+': '+str(best_clip_rate))
# clear cache
del model 
torch.cuda.empty_cache()
# reload weight
model = LlamaForCausalLM.from_pretrained(llm_path, revision="float16",torch_dtype=torch.float16).to(device)
f1_test,_ = test(model,test_data)
model_edit = LaserWrapper.get_edited_model(model=model,
                                                                lname = l_name,
                                                                lnum = l_num,
                                                                rate = best_clip_rate,
                                                                intervention="rank-reduction",
                                                                logger=logger,
                                                                in_place=True)
f1_test_svd,_ = test(model_edit,test_data) 
print('test f1:{} -> {}'.format(f1_test,f1_test_svd))
if search:
    with open(save_path+txtname, 'a+', encoding='utf-8') as f:
        f.write(('test f1:{} -> {}'.format(f1_test,f1_test_svd)))