# for question answering datasets

def sample_data_drop(dataset,seed=42, actual_sample_size = 2000, demo_size = 5):
    sample0 = dataset.shuffle(seed=seed)
    idxs = [i for i in range(len(sample0))]
    eval_id = idxs[:actual_sample_size]
    demo_id = idxs[actual_sample_size:actual_sample_size+demo_size]
    test_id = idxs[actual_sample_size+demo_size:]
    demo_sample = sample0.select(demo_id)
    eval_sample = sample0.select(eval_id)
    test_sample = sample0.select(test_id)
    return demo_sample,eval_sample,test_sample

def sample_data_squad(dataset,seed=42, actual_sample_size = 2000, demo_size = 5):
    sample0 = dataset.shuffle(seed=seed)
    idxs = [i for i in range(len(sample0))]
    eval_id = idxs[:actual_sample_size]
    demo_id = idxs[actual_sample_size:actual_sample_size+demo_size]
    test_id = idxs[actual_sample_size+demo_size:]
    demo_sample = sample0.select(demo_id)
    eval_sample = sample0.select(eval_id)
    test_sample = sample0.select(test_id)
    return demo_sample,eval_sample,test_sample

def cat_demo_drop(demo_sample,eval_sample,task_name, demo_shot=5):
    # prompt = f"passage: {context}\nquestion: {question}\nanswer:"
    assert task_name == "drop"
    new_data=[]
    for i in range(len(eval_sample)):
        item = dict()
        demo = demo_sample
        item['passage'] = eval_sample[i]['passage']
        item['question'] = eval_sample[i]['question']
        item['answers'] = eval_sample[i]['answers_spans']['spans']
        demo_sentence = ''
        for i in range(demo_shot):
            context = demo[i]['passage']
            question = demo[i]['question']
            answer = demo[i]['answers_spans']['spans'][0]
            demo_sentence += f"passage: {context}\nquestion: {question}\nanswer:{answer}\n"
        item['demo_sentence'] = demo_sentence
        new_data.append(item)
    return new_data

def cat_demo_squad(demo_sample,eval_sample,task_name, demo_shot=5):
    # prompt = f"title: {title}\ncontext: {context}\nquestion: {question}\nanswer:"
    assert task_name == "squad"
    new_data=[]
    for i in range(len(eval_sample)):
        item = dict()
        demo = demo_sample
        item['title'] = eval_sample[i]['title']
        item['context'] = eval_sample[i]['context']
        item['question'] = eval_sample[i]['question']
        item['answers'] = eval_sample[i]['answers']['text']
        demo_sentence = ''
        for i in range(demo_shot):
            title = demo[i]['title']
            context = demo[i]['context']
            question = demo[i]['question']
            answer = demo[i]['answers']['text'][0]
            demo_sentence += f"title: {title}\ncontext: {context}\nquestion: {question}\nanswer:{answer}\n"
        item['demo_sentence'] = demo_sentence
        new_data.append(item)
    return new_data