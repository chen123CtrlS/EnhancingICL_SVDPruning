from datasets import load_dataset
import pickle

def load_dataset_train_and_test(task_name,local_data_path=r''):
    """
    If there is no test data(or test data without label), we use validation for test.
    """
    dataset = None
    try:
        with open(local_data_path+'{}.pkl'.format(task_name), 'rb') as file:
            dataset = pickle.load(file)
    except:
        # classification
        if task_name == 'sst2':
            dataset = load_dataset('sst2', split=['train', 'validation'])
            # for i, _ in enumerate(dataset):
            #     dataset[i] = dataset[i].rename_column('sentence', 'text')
            dataset = {'train': dataset[0], 'test': dataset[1]}
        elif task_name == 'agnews':
            dataset = load_dataset('ag_news', split=['train', 'test'])
            dataset = {'train': dataset[0], 'test': dataset[1]}
        elif task_name == 'emoc':
            dataset = load_dataset('emo', split=['train', 'test'])
            dataset = {'train': dataset[0], 'test': dataset[1]}
        elif task_name == 'mrpc':
            dataset = load_dataset("glue", "mrpc")
            
        elif task_name == 'boolq':
            dataset = load_dataset("boolq")
        elif task_name == 'multirc':
            dataset = load_dataset("super_glue", "multirc")
        elif task_name == 'cb':
            dataset = load_dataset("super_glue", "cb")
        elif task_name == 'wic':
            dataset = load_dataset("super_glue", "wic")
        elif task_name == 'wsc':
            dataset = load_dataset("super_glue", "wsc.fixed")
        elif task_name == 'rte':
            dataset = load_dataset("super_glue", "rte")
        # multiple-choice
        elif task_name == 'copa':
            dataset = load_dataset('super_glue', "copa")
        elif task_name == 'record':
            dataset = load_dataset("super_glue", "record")
        # question answering
        elif task_name == 'squad':
            dataset = load_dataset("squad")
        elif task_name == 'drop':
            dataset = load_dataset("drop")
        if dataset is None:
            raise NotImplementedError(f"task_name: {task_name}")
    if task_name in ['copa','boolq','multirc','cb','wic','wsc','record','rte','squad','drop']:
        dataset = {'train': dataset['train'], 'test': dataset['validation']}
    return dataset