import random
import pandas as pd
import torch
import numpy as np
import os
import json
from typing import *

def project_onto_direction(H, direction):
    """Project matrix H (n, d_1) onto direction vector (d_2,)"""
    # Calculate the magnitude of the direction vector
     # Ensure H and direction are on the same device (CPU or GPU)
    if type(direction) != torch.Tensor:
        H = torch.Tensor(H).cuda()
    if type(direction) != torch.Tensor:
        direction = torch.Tensor(direction)
        direction = direction.to(H.device)
    mag = torch.norm(direction)
    assert not torch.isinf(mag).any()
    # Calculate the projection
    projection = H.matmul(direction) / mag
    return projection



def recenter(x, mean=None):
    x = torch.Tensor(x).cuda()
    if mean is None:
        mean = torch.mean(x,axis=0,keepdims=True).cuda()
    else:
        mean = torch.Tensor(mean).cuda()
    return x - mean


def honesty_function_dataset(data_path: str, tokenizer, user_tag: str = "", assistant_tag: str = "", seed: int = 0) -> (list, list):
    """
    Processes data to create training and testing datasets based on honesty.

    Args:
    - data_path (str): Path to the CSV containing the data.
    - tokenizer (PreTrainedTokenizer): Tokenizer to tokenize statements.
    - user_tag (str): Instruction template.
    - assistant_tag (str): Instruction template user tag.
    - seed (int): Random seed for reproducibility.

    Returns:
    - Tuple containing train and test data.
    """

    # Setting the seed for reproducibility
    random.seed(seed)

    # Load the data
    df = pd.read_csv(data_path)
    true_statements = df[df['label'] == 1]['statement'].values.tolist()
    false_statements = df[df['label'] == 0]['statement'].values.tolist()

    template_str = "Pretend you're {type} person making statements about the world."
    honest_statements = []
    untruthful_statements = []

    # Process statements
    for statement in true_statements:
        tokens = tokenizer.tokenize(statement)
        
        for idx in range(1, len(tokens) - 5):
            truncated_tokens = tokens[:idx]
            truncated_statement = tokenizer.convert_tokens_to_string(truncated_tokens)

            honest_statements.append(f"{user_tag} {template_str.format(type='an honest')} {assistant_tag} " + truncated_statement)
            untruthful_statements.append(f"{user_tag} {template_str.format(type='an untruthful')} {assistant_tag} " + truncated_statement)

    # Create training data
    ntrain = 512
    combined_data = [[honest, untruthful] for honest, untruthful in zip(honest_statements, untruthful_statements)]
    train_data = combined_data[:ntrain]

    train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append([s == true_s for s in d])
    
    train_data = np.concatenate(train_data).tolist()

    # Create test data
    reshaped_data = np.array([[honest, untruthful] for honest, untruthful in zip(honest_statements[:-1], untruthful_statements[1:])]).flatten()
    test_data = reshaped_data[ntrain:ntrain*2].tolist()

    print(f"Train data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")

    return {
        'train': {'data': train_data, 'labels': train_labels},
        'test': {'data': test_data, 'labels': [[1,0]] * len(test_data)}
    }

def primary_emotions_concept_dataset(data_dir, user_tag='', assistant_tag='', seed=0):
    random.seed(0)

    template_str = '{user_tag} Consider the {emotion} of the following scenario:\nScenario: {scenario}\nAnswer: {assistant_tag} '
    emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
    raw_data = {}
    for emotion in emotions:
        with open(os.path.join(data_dir, f'{emotion}.json')) as file:
            # raw_data[emotion] = json.load(file)
            raw_data[emotion] = list(set(json.load(file)))[:100]

    formatted_data = {}
    for emotion in emotions:
        c_e, o_e = raw_data[emotion], np.concatenate([v for k,v in raw_data.items() if k != emotion])
        random.shuffle(o_e)

        data = [[c,o] for c,o in zip(c_e, o_e)]
        train_labels = []
        for d in data:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])
        
        data = np.concatenate(data).tolist()
        data_ = np.concatenate([[c,o] for c,o in zip(c_e, o_e)]).tolist()
        
        emotion_test_data = [template_str.format(emotion=emotion, scenario=d, user_tag=user_tag, assistant_tag=assistant_tag) for d in data_]
        emotion_train_data = [template_str.format(emotion=emotion, scenario=d, user_tag=user_tag, assistant_tag=assistant_tag) for d in data]

        formatted_data[emotion] = {
            'train': {'data': emotion_train_data, 'labels': train_labels},
            'test': {'data': emotion_test_data, 'labels': [[1,0]* len(emotion_test_data)]}
        }
    return formatted_data

def primary_emotions_function_dataset(data_dir, user_tag='', assistant_tag='', seed=0):
    random.seed(0)

    train_template_str = '{user_tag} Act as if you are extremely {emo}. {assistant_tag} {scenario}' 
    emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
    with open(os.path.join(data_dir, "all_truncated_outputs.json"), 'r') as file:
        all_truncated_outputs = json.load(file)
    
    emotions = ["happiness", "sadness", "anger", "fear", "disgust", "surprise"]
    emotions_adj = [
        ("joyful", "happy", "cheerful"), 
        ("sad", "depressed", "miserable"),
        ("angry", "furious", "irritated"),
        ("fearful", "scared", "frightened"),
        ("disgusted", "sicken", "revolted"), 
        ("surprised", "shocked", "astonished")
    ]
    emotions_adj_ant = [
        ("dejected", "unhappy", "dispirited"), 
        ("cheerful", "optimistic", "happy"),
        ("pleased", "calm", "peaceful"),
        ("fearless", "bold", "unafraid"),
        ("approved", "delighted", "satisfied"), 
        ("unimpressed", "indifferent", "bored")
    ]

    formatted_data = {}
    for emotion, emotion_adj, emotion_adj_ant in zip(emotions, emotions_adj, emotions_adj_ant):
        emotion_train_data_tmp = [[
            train_template_str.format(emo=np.random.choice(emotion_adj), scenario=s, user_tag=user_tag, assistant_tag=assistant_tag), 
            train_template_str.format(emo=np.random.choice(emotion_adj_ant), scenario=s, user_tag=user_tag, assistant_tag=assistant_tag)
        ] for s in all_truncated_outputs]
        
        train_labels = []
        for d in emotion_train_data_tmp:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])

        emotion_train_data = np.concatenate(emotion_train_data_tmp).tolist()

        formatted_data[emotion] = {
            'train': {'data': emotion_train_data, 'labels': train_labels},
        }
    return formatted_data


# --------------Our custom data function--------------
def custom_concept_dataset(concept_data_path, 
                           user_tag: str = "", 
                           assistant_tag: str = "", 
                           seed: int = 0,
                           ) -> (list, list):
    # input should be a csv file list and it's label.
    
    random.seed(0)
    
    def get_template_str(is_role_play):
        if is_role_play:
            template_str = "{user_tag} You and your friend are role-playing. You are taking on the role of {emotion}. Answer question based on {emotion}'s Tone. Example: {scenario} {assistant_tag}"
        else:
            template_str = '{user_tag} Consider the {emotion} of the following scenario and answer question with a tone of {emotion}. Example scenario: {scenario} {assistant_tag}'
        return template_str
    
    csv_file_paths = [os.path.join(concept_data_path, i) for i in os.listdir(concept_data_path)]
    emotions = []
    random.shuffle(csv_file_paths)
    raw_data = {}
    is_role_play_list = []
    for csv_file_path in csv_file_paths:
        emotion = csv_file_path.split('/')[-1].split('.')[0].split('_')[0]
        emotions.append(emotion)
        is_role_play = csv_file_path.split('/')[-1].split('.')[0].split('_')[1] == 'rp'
        is_role_play_list.append(is_role_play)
        raw_data[emotion] = list(set(pd.read_csv(csv_file_path)['text']))[:512]
        cur_df = pd.read_csv(csv_file_path)['text']


    formatted_data = {}
    for emo_idx, emotion in enumerate(emotions):
        is_role_play = is_role_play_list[emo_idx]
        c_e, o_e = raw_data[emotion], np.concatenate([v for k,v in raw_data.items() if k != emotion])
        random.shuffle(o_e)
        template_str = get_template_str(is_role_play)
        data = [[c,o] for c,o in zip(c_e, o_e)]
        train_labels = []
        for d in data:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])
        
        data = np.concatenate(data).tolist()
        data_ = np.concatenate([[c,o] for c,o in zip(c_e, o_e)]).tolist()
        
        emotion_test_data = [template_str.format(emotion=emotion, scenario=d, user_tag=user_tag, assistant_tag=assistant_tag) for d in data_]
        emotion_train_data = [template_str.format(emotion=emotion, scenario=d, user_tag=user_tag, assistant_tag=assistant_tag) for d in data]

        formatted_data[emotion] = {
            'train': {'data': emotion_train_data, 'labels': train_labels},
            'test': {'data': emotion_test_data, 'labels': [[1,0]* len(emotion_test_data)]}
        }
    return formatted_data
    
def custom_function_dataset(data_path: str, 
                            positive_direction_name,
                            negative_direction_name,
                            tokenizer, 
                            user_tag: str = "", 
                            assistant_tag: str = "", 
                            seed: int = 0,
                            is_role_play: Optional[bool] = None,
                            ) -> (list, list):
    """
    Processes data to create training and testing datasets based on honesty.

    Args:
    - data_path (str): Path to the CSV containing the data.
    - tokenizer (PreTrainedTokenizer): Tokenizer to tokenize statements.
    - user_tag (str): Instruction template.
    - assistant_tag (str): Instruction template user tag.
    - seed (int): Random seed for reproducibility.

    Returns:
    - Tuple containing train and test data.
    """
    # Setting the seed for reproducibility
    if is_role_play is None:
        is_role_play = data_path.split('/')[-1].split('.')[0].split('_')[-1] == 'rp' # rp for roll play, nrp for non-roll-play
    random.seed(seed)
    def get_template_str(ai_mode:str, truncat_statement:str):
        if is_role_play:
            template_str = f"You and your friend are role-playing. You are taking on the role of {ai_mode}."
        else:
            template_str = f"Act as if you are extremely {ai_mode}."
        return f"{user_tag} {template_str} {assistant_tag} {truncat_statement}"

    # Load the data
    df = pd.read_csv(data_path)
    df = preprocess(df)
    # some labels are negative and some labes are positive
    positive_statements = df[df['label'] == 1]['statement'].values.tolist()
    negative_statements = df[df['label'] == 0]['statement'].values.tolist()

    positive_training_data = []
    negative_training_data = []

    # Process statements
    for statement in positive_statements:
        tokens = tokenizer.tokenize(statement)
        
        for idx in range(1, len(tokens) - 5):
            truncated_tokens = tokens[:idx]
            
            truncated_statement = tokenizer.convert_tokens_to_string(truncated_tokens)
            
            positive_train_str = get_template_str(positive_direction_name, truncated_statement)
            negative_train_str = get_template_str(negative_direction_name, truncated_statement)

            positive_training_data.append(positive_train_str)
            negative_training_data.append(negative_train_str)

    # Create training data
    ntrain = 2048
    combined_data = [[honest, untruthful] for honest, untruthful in zip(positive_training_data, negative_training_data)]
    train_data = combined_data[:ntrain]

    train_labels = []
    for d in train_data:
        true_s = d[0]
        random.shuffle(d)
        train_labels.append([s == true_s for s in d])
    
    train_data = np.concatenate(train_data).tolist()

    # Create test data
    reshaped_data = np.array([[honest, untruthful] for honest, untruthful in zip(positive_training_data[:-1], negative_training_data[1:])]).flatten()
    test_data = reshaped_data[ntrain:ntrain*2].tolist()

    print(f"Train data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")

    return {
        'train': {'data': train_data, 'labels': train_labels},
        'test': {'data': test_data, 'labels': [[1,0]] * len(test_data)}
    }
def preprocess(cur_df: pd.DataFrame) -> pd.DataFrame:
    '''
    clean label and content of the csv
    '''
    label_col = cur_df.select_dtypes(include=['number'])
    assert label_col.shape[1] == 1, f'''Only one column could have number entries'''
    label_list = list(label_col[list(label_col.keys())[0]].unique())
    assert 1 in label_list, f'''Label must contain 1 and 1 should be positive label'''
    label_list.remove(1)
    label_mapping = {1:1, 0:label_list[0]}
    new_df = {}
    new_df.update({'statement':[], 'label':[]})
    for index, row in cur_df.iterrows():
        keys = list(row.keys())
        have_str = False
        have_int = False
        assert len(keys) == 2, f'''Need each row have exactly 2 element but we get {len(keys)} for row: {row}'''
        for key in keys:
            if isinstance(row[key], str):
                new_df['statement'].append(row[key].replace('"',''))
                have_str = True
            elif isinstance(row[key], int):
                new_df['label'].append(row[key])
                have_int = True
            else:
                assert False, f'''{row[key]} is an unsupported type: {type(row[key])}'''
        assert have_str and have_int, f'''Need exactly have one int and one string entries'''
    return pd.DataFrame(new_df)