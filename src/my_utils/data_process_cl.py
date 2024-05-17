# for classification datasets
def sample_data_sst2(dataset,seed=42,actual_sample_size = 2000, demo_size = 5):
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


def sample_data_emo(dataset,seed=42,actual_sample_size = 2000, demo_size = 5):
    sample0 = dataset.shuffle(seed=seed)
    p_id = []
    n_id = []
    for idx,i in enumerate(sample0['label'][:]):
        if i == 0:
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

def sample_data_agnews(dataset,seed=42,actual_sample_size = 1000, demo_size = 2):
    sample0 = dataset.shuffle(seed=seed)
    id_list = [[],[],[],[]]
    for idx,i in enumerate(sample0['label'][:]):
        id_list[i].append(idx)
    demo_id0 = id_list[0][actual_sample_size:actual_sample_size+ demo_size]
    demo_id1 = id_list[1][actual_sample_size:actual_sample_size+ demo_size]
    demo_id2 = id_list[2][actual_sample_size:actual_sample_size+ demo_size]
    demo_id3 = id_list[3][actual_sample_size:actual_sample_size+ demo_size]
    demo_id = []
    for i in range(demo_size):
        demo_id.append(demo_id0[i])
        demo_id.append(demo_id1[i])
        demo_id.append(demo_id2[i])
        demo_id.append(demo_id3[i])
    demo_sample = sample0.select(demo_id)
    eval_sample = sample0.select(id_list[0][:actual_sample_size]+id_list[1][:actual_sample_size]\
                                 +id_list[2][:actual_sample_size]+id_list[3][:actual_sample_size]
                                 )
    return demo_sample,eval_sample

def sample_data_mrpc(dataset,seed=42,actual_sample_size = 1000, demo_size = 5):
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


def cat_demo(demo_sample,eval_sample,task_name, demo_shot=10):
    # for sst2
    new_data=[]
    for i in range(len(eval_sample)):
        item = dict()
        demo = demo_sample
        item['text'] = eval_sample[i]['sentence']
        item['label'] = eval_sample[i]['label']
        demo_sentence = ''
        for i in range(demo_shot):
            lll = demo[i]['label']
            if task_name == 'sst2':
                iii = lll
            else:
                raise NotImplementedError(f"task_name: {task_name}")
            text = demo[i]['sentence']
            if iii == 1: 
                demo_sentence += 'text: ' + text +'\n sentiment : pos \n'
            else:
                demo_sentence += 'text: ' + text +'\n sentiment : neg \n'
        item['demo_sentence'] = demo_sentence
        new_data.append(item)
    return new_data


def cat_demo_emo(demo_sample,eval_sample,task_name, demo_shot=10):
    # for emo
    new_data=[]
    for i in range(len(eval_sample)):
        item = dict()
        demo = demo_sample
        item['text'] = eval_sample[i]['text']
        item['label'] = eval_sample[i]['label']
        demo_sentence = ''
        for i in range(demo_shot):
            lll = demo[i]['label']
            if task_name  == 'emoc':
                if lll == 0:
                    iii = 1
                else:
                    iii = 0
            else:
                raise NotImplementedError(f"task_name: {task_name}")
            text = demo[i]['text']
            if iii == 1: 
                demo_sentence += 'text: ' + text +'\n sentiment : pos \n'
            else:
                demo_sentence += 'text: ' + text +'\n sentiment : neg \n'
        item['demo_sentence'] = demo_sentence
        new_data.append(item)
    return new_data


def cat_demo_agnews(demo_sample,eval_sample,task_name, demo_shot=10):
    new_data=[]
    for i in range(len(eval_sample)):
        item = dict()
        demo = demo_sample
        item['text'] = eval_sample[i]['text']
        item['label'] = eval_sample[i]['label']
        demo_sentence = ''
        for i in range(demo_shot):
            lll = demo[i]['label']
            if task_name == 'agnews':
                iii = lll
            else:
                raise NotImplementedError(f"task_name: {task_name}")
            text = demo[i]['text']
            # World, Sports, Business, Sci/Tech(science)
            if iii == 0: 
                demo_sentence += 'text: ' + text +'\n sentiment : world \n'
            elif iii == 1:
                demo_sentence += 'text: ' + text +'\n sentiment : sports \n'
            elif iii == 2:
                demo_sentence += 'text: ' + text +'\n sentiment : business \n'
            else:
                demo_sentence += 'text: ' + text +'\n sentiment : science \n'
        item['demo_sentence'] = demo_sentence
        new_data.append(item)
    return new_data


def cat_demo_mrpc(demo_sample,eval_sample,task_name, demo_shot=10):
    assert task_name == "mrpc"
    new_data=[]
    for i in range(len(eval_sample)):
        item = dict()
        demo = demo_sample
        item['text'] = 'sentence1: '+eval_sample[i]['sentence1']+ ' \n sentence2: '+eval_sample[i]['sentence2']
        item['label'] = eval_sample[i]['label']
        demo_sentence = ''
        for i in range(demo_shot):
            iii = demo[i]['label']
            s1 = demo[i]['sentence1']
            s2 = demo[i]['sentence2']
            if iii == 1: #equal
                demo_sentence += 'sentence1: ' + s1 +' \n sentence2: '+ s2+ '\n label : true \n'
            else:
                demo_sentence += 'sentence1: ' + s1 +' \n sentence2: '+ s2+ '\n label : false \n'
        item['demo_sentence'] = demo_sentence
        new_data.append(item)
    return new_data


# rte dataset
def sample_data_rte(dataset,seed=42,actual_sample_size = 200, demo_size = 5):
    sample0 = dataset.shuffle(seed=seed)
    p_id = []
    n_id = []
    for idx,i in enumerate(sample0['label'][:]):
        if i == 0: #Yes
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

def cat_demo_rte(demo_sample,eval_sample,task_name, demo_shot=10):
    new_data=[]
    for i in range(len(eval_sample)):
        item = dict()
        demo = demo_sample
        item['premise'] =  eval_sample[i]['premise']
        item['hypothesis'] = eval_sample[i]['hypothesis']
        item['label'] = eval_sample[i]['label']
        demo_sentence = ''
        for i in range(demo_shot):
            premise = demo[i]['premise']
            hypothesis = demo[i]['hypothesis']
            label = demo[i]['label']
            if label == 0:
                choice = 'Yes'
            else:
                choice = 'No'
            demo_sentence += f"{premise} Does this mean that {hypothesis} is true? select Yes or No? {choice}\n"
        item['demo_sentence'] = demo_sentence
        new_data.append(item)
    return new_data

def sample_data_cb(dataset,seed=42,actual_sample_size = 60, demo_size = 5):
    sample0 = dataset.shuffle(seed=seed)
    p_id = []
    n_id = []
    m_id = []
    for idx,i in enumerate(sample0['label'][:]):
        if i == 0: #Yes
            p_id.append(idx)
        elif i == 1: #No
            m_id.append(idx)
        else:
            n_id.append(idx)
    p_demo_id = p_id[:demo_size]
    n_demo_id = n_id[:demo_size]
    m_demo_id = m_id[:demo_size]
    demo_id = []
    for i in range(demo_size):
        demo_id.append(p_demo_id[i])
        demo_id.append(n_demo_id[i])
        demo_id.append(m_demo_id[i])
    demo_sample = sample0.select(demo_id)
    eval_sample = sample0.select(p_id[demo_size:demo_size+actual_sample_size]+n_id[demo_size:demo_size+actual_sample_size]+m_id[demo_size:demo_size+actual_sample_size])
    test_sample = sample0.select(p_id[actual_sample_size+ demo_size:]+n_id[actual_sample_size+ demo_size:]+m_id[actual_sample_size+ demo_size:])
    return demo_sample,eval_sample,test_sample

def cat_demo_cb(demo_sample,eval_sample,task_name, demo_shot=15):
    assert task_name == 'cb'
    new_data=[]
    for i in range(len(eval_sample)):
        item = dict()
        demo = demo_sample
        item['premise'] =  eval_sample[i]['premise']
        item['hypothesis'] = eval_sample[i]['hypothesis']
        item['label'] = eval_sample[i]['label']
        demo_sentence = ''
        for i in range(demo_shot):
            premise = demo[i]['premise']
            hypothesis = demo[i]['hypothesis']
            label = demo[i]['label']
            if label == 0:
                choice = 'Yes'
            elif label == 1:
                choice = 'No'
            else:
                choice = 'Mi'
            demo_sentence += f"Suppose {premise} Can we infer that '{hypothesis}'? Yes, No, or Maybe? {choice}\n"
        item['demo_sentence'] = demo_sentence
        new_data.append(item)
    return new_data