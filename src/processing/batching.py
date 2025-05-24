import json
import random
import os
from pathlib import Path
import concurrent.futures

current_dir = os.path.dirname(os.path.abspath(__file__))

def count_words(message):
    count = 0
    for msg in message:
        if 'content' in msg:
            count += len(msg['content'].split())
    return count


def chunk_messages(messages, max_length=3000, max_messages=2):
    # Sort messages by length in descending order (helps with efficient packing)
    sorted_msgs = sorted(messages, key=lambda x: x['word_count'], reverse=True)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Try to pack messages into chunks
    for msg in sorted_msgs:
        # Check both word count limit and message count limit
        if current_length + msg['word_count'] <= max_length and len(current_chunk) < max_messages:
            # Message fits in current chunk
            current_chunk.append(msg)
            current_length += msg['word_count']
        else:
            # Message doesn't fit or chunk is at max message count, start a new chunk
            if current_chunk:  # Only append if not empty
                chunks.append(current_chunk)
            current_chunk = [msg]
            current_length = msg['word_count']
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def concate_chunk(chunk):
    # Concatenate all messages in the chunk
    concatenated = []
    system_prompt = chunk[0]['messages'][0]['content']
    concatenated.append({
        'role': 'system',
        'content': system_prompt
    })
    
    for messages in chunk:
        for message in messages['messages']:
            if message['role'] == 'user':
                concatenated.append({
                    'role': 'user',
                    'content': message['content']
                })
            elif message['role'] == 'assistant':
                concatenated.append({
                    'role': 'assistant',
                    'content': message['content']
                })
        
    return concatenated



def _load_data_from_path(data_files):
    """
    This function loads data from the specified path and returns a list of messages.
    """
    data = []
    for file in data_files:
        if file.endswith('.json'):
            with open(file, 'r') as f:
                messages = json.load(f)
                for message in messages:
                    message['word_count'] = count_words(message['messages'])
                    data.append(message)

        elif file.endswith('.jsonl'):
            with open(file, 'r') as f:
                for line in f:
                    message = json.loads(line)
                    message['word_count'] = count_words([message])
                    data.append(message)

    return data



def dynamic_batching(data_files, max_length=1000, max_messages=2):
    """
    This function takes in three lists of messages (long, short, and news_data) and creates batches of messages
    that are grouped together based on their word count and message count.
    """

    # Load data from files
    print("Loading data from files...")
    data = _load_data_from_path(data_files)
    print(f"Loaded {len(data)} messages from {len(data_files)} files.")

    # Separate messages into categories
    short = [msg for msg in data if msg['word_count'] < max_length]
    long = [msg for msg in data if msg['word_count'] >= max_length]

    # Chunk messages
    print("Number of short messages:", len(short))
    print("Number of long messages:", len(long))
    short_chunks = chunk_messages(short, max_length=max_length, max_messages=max_messages)
    print(f"Number of short chunks: {len(short_chunks)}")

    short_msg = []
    for chunk in short_chunks:
        msg = concate_chunk(chunk)
        short_msg.append({
            'messages': msg,
            'index': len(short_msg),
            'word_count': count_words(msg)
        })

    

    BS = 16

    group_bs = []

    for i in range(0, len(short_msg) - BS, BS):
        group = short_msg[i:i+BS]
        group_bs.append(group)

    for i in range(0, len(long) - BS, BS):
        group = long[i:i+BS]
        group_bs.append(group)

    random.shuffle(group_bs)

    data = []
    for bs in group_bs:
        for msg in bs:
            data.append(msg)

    print(f"Number of messages after batching: {len(data)}")

    new_data_files = []
    chunk_length = 50000



    def write_chunk(chunk_idx, chunk_data):
        file_path = os.path.join(current_dir, f"../../LLaMA-Factory/data/batch_{chunk_idx + 1}.jsonl")
        with open(file_path, 'w') as f:
            for item in chunk_data:
                f.write(json.dumps(item) + '\n')
        return file_path

    # Split data into chunks
    chunks = [(i//chunk_length, data[i:i+chunk_length]) 
              for i in range(0, len(data), chunk_length)]

    # # Process chunks using threads instead of processes
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     new_data_files = list(executor.map(lambda x: write_chunk(x[0], x[1]), chunks))

    for i, chunk in chunks:
        print(f"Writing chunk {i + 1} with {len(chunk)} messages...")
        file_path = write_chunk(i, chunk)
        print(f"Chunk {i + 1} written to {file_path}")
        new_data_files.append(file_path)

    # Delete the original files
    for file in data_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Deleted original file: {file}")
    
    return new_data_files