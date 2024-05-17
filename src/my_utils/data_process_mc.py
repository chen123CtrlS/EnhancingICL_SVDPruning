# for multiple-choice datasets

def sample_data_copa(dataset,seed=42, actual_sample_size = 100, demo_size = 5):
    sample0 = dataset.shuffle(seed=seed)
    p_id = []
    n_id = []
    for idx,i in enumerate(sample0['label'][:]):
        if i == 1:
            p_id.append(idx)
        else:
            n_id.append(idx)
    p_demo_id = p_id[actual_sample_size:actual_sample_size+ demo_size]
    n_demo_id = n_id[actual_sample_size:actual_sample_size+ demo_size]
    demo_id = []
    for i in range(demo_size):
        demo_id.append(p_demo_id[i])
        demo_id.append(n_demo_id[i])
    demo_sample = sample0.select(demo_id)
    eval_sample = sample0.select(p_id[:actual_sample_size]+n_id[:actual_sample_size])
    test_sample = sample0.select(p_id[actual_sample_size+ demo_size:]+n_id[actual_sample_size+ demo_size:])
    return demo_sample,eval_sample,test_sample


def cat_demo_copa(demo_sample,eval_sample,task_name, demo_shot=10):
    assert task_name == "copa"
    new_data=[]
    for i in range(len(eval_sample)):
        item = dict()
        demo = demo_sample
        item['premise'] = eval_sample[i]['premise']
        item['question'] = eval_sample[i]['question']
        item['choice1'] = eval_sample[i]['choice1']
        item['choice2'] = eval_sample[i]['choice2']
        item['label'] = eval_sample[i]['label']
        demo_sentence = ''
        for i in range(demo_shot):
            iii = demo[i]['label']
            premise = demo[i]['premise']
            question = demo[i]['question']
            choice1 = demo[i]['choice1']
            choice2 = demo[i]['choice2']
            if iii == 0: #so
                demo_sentence += f"{premise}\n{question}\n{choice1}\n"
            else: #because
                demo_sentence += f"{premise}\n{question}\n{choice2}\n"
        item['demo_sentence'] = demo_sentence
        new_data.append(item)
    return new_data