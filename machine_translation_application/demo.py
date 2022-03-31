import tensorflow as tf
tf.enable_eager_execution()
import tensorflow_datasets as tfds
import os

def split_task_name(task_name):
    task_name_split = task_name.split('_to_')
    return task_name_split[0], task_name_split[1]


# Get dataset
def get_ted_hrlr_translate_dataset(task_name="pt_to_en", BATCH_SIZE=64, MAX_LENGTH=40, BUFFER_SIZE=20000, 
                                    languageA_target_vovab_size=2**13, languageB_target_vovab_size=2**13):
    """
    task_name: string
    BATCH_SIZE: int
    MAX_LENGTH: int
    BUFFER_SIZE: int
    languageA_target_vacab_size: int
    languageB_target_vacab_size: int
    """
    task_name_prefix = 'ted_hrlr_translate'
    task_name_list = ['az_to_en', 'az_tr_to_en', 'be_to_en',
                      'be_ru_to_en', 'es_to_pt', 'fr_to_pt',
                      'gl_to_en', 'gl_pt_to_en', 'he_to_pt',
                      'it_to_pt', 'pt_to_en', 'ru_to_en',
                      'ru_to_pt', 'tr_to_en']
    if task_name not in task_name_list:
        raise ValueError(f'Choose task_name from {task_name_list}')

    complete_task_name = task_name_prefix + '/' + task_name
    # Get the dataset
    example, metadata = tfds.load(complete_task_name, with_info=True, as_supervised=True)
    train_examples, val_examples = example['train'], example['validation']

    # make dir to store data
    if not os.path.exists(complete_task_name):
        os.makedirs(complete_task_name)

    # load data and encode the string as int
    tokenizer_languageA_path = os.path.join(complete_task_name, split_task_name(task_name)[0])
    tokenizer_languageA_complete_path = tokenizer_languageA_path + ".subwords"
    if not os.path.exists(tokenizer_languageA_complete_path):
        tokenizer_languageA = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (languageA.numpy() for languageA, _ in train_examples), 
            trarget_vocab_size=languageA_target_vovab_size)
        tokenizer_languageA.save_to_file(tokenizer_languageA_path)
    else:
        tokenizer_languageA = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizer_languageA_path)

    tokenizer_languageB_path = os.path.join(complete_task_name, split_task_name(task_name)[1])
    tokenizer_languageB_complete_path = tokenizer_languageB_path + ".subwords"
    if not os.path.exists(tokenizer_languageB_complete_path):
        tokenizer_languageB = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (languageB.numpy() for _, languageB in train_examples), 
            trarget_vocab_size=languageB_target_vovab_size)
        tokenizer_languageB.save_to_file(tokenizer_languageB_path)
    else:
        tokenizer_languageB = tfds.features.text.SubwordTextEncoder.load_from_file(tokenizer_languageB_path)

    print(tokenizer_languageA)

    def encode(lang1, lang2):
        lang1 = [tokenizer_languageA.vocab_size]


    return


if __name__ == "__main__":
    get_ted_hrlr_translate_dataset()