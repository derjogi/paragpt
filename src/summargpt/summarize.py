from datetime import datetime
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
import networkx as nx
from networkx.algorithms import community

from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
import re
from os.path import splitext, exists

os.environ["OPENAI_API_KEY"] = '...'
vtt_path = "transcript.vtt"
txt_path = "transcript.txt"


def save_to_file(content, filename: str = None) -> str:
    # set default values
    if not filename:
        filename = "out.txt"
        i = 0
        while exists(filename):
            i += 1
            filename = "out_%s.txt" % i

    with open(filename, "w+", encoding="utf-8") as fp:
        if isinstance(content, dict):  # Check if content is a dict (JSON)
            json.dump(content, fp, indent=4)
        elif isinstance(content, pd.DataFrame):  # Check if content is a DataFrame
            content.to_csv(fp, index=False)
        elif isinstance(content, str):  # Check if content is a string
            fp.write(content)
        else:
            print(f"Unsupported data type: {type(content)}")

    return filename

def read_from_file(filename: str, load_as_json: bool = True) -> str:
    try:
        # Try to load as JSON
        with open(filename, "r", encoding="utf-8") as fp:
            return json.load(fp)
    except json.JSONDecodeError:
        pass

    try:
        # Try to load as DataFrame
        return pd.read_csv(filename)
    except pd.errors.ParserError:
        pass

    try:
        # Try to load as string
        with open(filename, "r", encoding="utf-8") as fp:
            return fp.read()
    except Exception as e:
        print(f"Failed to read file: {e}")

    return None

def main(convo_file: str, chunks_file: str = None, stage1_file: str = None, stage2_file: str = None):
    if chunks_file == None or not exists(chunks_file):
        segments = create_segments(convo_file)
        sentences = create_sentences(segments, MIN_WORDS=20, MAX_WORDS=80)
        chunks = create_chunks(sentences, CHUNK_LENGTH=5, STRIDE=1)
        chunks_text = [chunk['text'] for chunk in chunks]
        save_to_file(chunks_text, "chunks.out")
    else :
        chunks_text = read_from_file(chunks_file, False)

    # Run Stage 1 Summarizing

    ### Stage 1: Getting Chunk Summaries
    if stage1_file == None or not exists(stage1_file):
        stage_1_outputs = summarize_stage_1(chunks_text)['stage_1_outputs']
        save_to_file(stage_1_outputs, "stage1.out")
    else:
        stage_1_outputs = read_from_file(stage1_file)
    # Split the titles and summaries
    stage_1_summaries = [e['summary'] for e in stage_1_outputs]
    stage_1_titles = [e['title'] for e in stage_1_outputs]
    num_1_chunks = len(stage_1_summaries)

    # Use OpenAI to embed the summaries and titles. Size of _embeds: (num_chunks x 1536)
    openai_embed = OpenAIEmbeddings()

    summary_embed_file = "embeddings_summaries.txt"
    if exists(summary_embed_file):
        summary_embeds = read_from_file(summary_embed_file)
    else:
        summary_embeds = np.array(openai_embed.embed_documents(stage_1_summaries))

    title_embed_file = "embeddings_titles.txt"
    if exists(title_embed_file):
        title_embeds = read_from_file(title_embed_file)
    else:
        title_embeds = np.array(openai_embed.embed_documents(stage_1_titles))

    save_to_file(summary_embeds, summary_embed_file)
    save_to_file(title_embeds, title_embed_file)

    # Get similarity matrix between the embeddings of the chunk summaries
    summary_similarity_matrix = np.zeros((num_1_chunks, num_1_chunks))
    summary_similarity_matrix[:] = np.nan

    for row in range(num_1_chunks):
        for col in range(row, num_1_chunks):
            # Calculate cosine similarity between the two vectors
            similarity = 1 - cosine(summary_embeds[row], summary_embeds[col])
            summary_similarity_matrix[row, col] = similarity
            summary_similarity_matrix[col, row] = similarity

    # Draw a heatmap with the summary_similarity_matrix
    plt.figure()
    # Color scheme blues
    plt.imshow(summary_similarity_matrix, cmap='Blues')

    # Set num_topics to be 1/4 of the number of chunks, or 8, which ever is smaller
    num_topics = min(int(num_1_chunks / 4), 8)
    topics_out = get_topics(summary_similarity_matrix, num_topics=num_topics, bonus_constant=0.2)
    chunk_topics = topics_out['chunk_topics']
    topics = topics_out['topics']

    # Plot a heatmap of this array
    plt.figure(figsize=(10, 4))
    plt.imshow(np.array(chunk_topics).reshape(1, -1), cmap='tab20')
    # Draw vertical black lines for every 1 of the x-axis
    for i in range(1, len(chunk_topics)):
        plt.axvline(x=i - 0.5, color='black', linewidth=0.5)

    ### Stage 2 Summaries
    # Query GPT-3 to get a summarized title for each topic_data
    out = summarize_stage_2(stage_1_outputs, topics, summary_num_words=250)
    stage_2_outputs = out['stage_2_outputs']
    stage_2_titles = [e['title'] for e in stage_2_outputs]
    stage_2_summaries = [e['summary'] for e in stage_2_outputs]
    final_summary = out['final_summary']

    print(stage_2_outputs)
    print(final_summary)
    save_to_file(out, "stage2.out")

def create_segments(txt_path: str):
    with open(txt_path, 'r') as f:
        txt = f.read()
    # Get segments from txt by splitting on Newlines
    segments =  txt.split('\n')
    # Further split by . -> segments[sentences[]]
    segments = [segment.split('.') for segment in segments]
    # Put the . back in
    segments = [sentence + '.' for segment in segments for sentence in segment]
    # Further split by comma -> segments[sentences[items[]]]
    segments = [sentence.split(',') for sentence in segments]
    # Flatten
    segments = [item for sentence in segments for item in sentence]
    return segments


def create_sentences(segments, MIN_WORDS, MAX_WORDS):

    # Combine the non-sentences together
    sentences = []

    is_new_sentence = True
    sentence_length = 0
    sentence_num = 0
    sentence_segments = []

    for i in range(len(segments)):
        if is_new_sentence == True:
            is_new_sentence = False
        # Append the segment
        sentence_segments.append(segments[i])
        segment_words = segments[i].split(' ')
        sentence_length += len(segment_words)

        # If exceed MAX_WORDS, then stop at the end of the segment
        # Only consider it a sentence if the length is at least MIN_WORDS
        if (sentence_length >= MIN_WORDS and segments[i][-1] == '.') or sentence_length >= MAX_WORDS:
            sentence = ' '.join(sentence_segments)
            sentences.append({
                'sentence_num': sentence_num,
                'text': sentence,
                'sentence_length': sentence_length
            })
            # Reset
            is_new_sentence = True
            sentence_length = 0
            sentence_segments = []
            sentence_num += 1

    return sentences


def create_chunks(sentences, CHUNK_LENGTH, STRIDE):

    sentences_df = pd.DataFrame(sentences)

    chunks = []
    for i in range(0, len(sentences_df), (CHUNK_LENGTH - STRIDE)):
        chunk = sentences_df.iloc[i:i+CHUNK_LENGTH]
        chunk_text = ' '.join(chunk['text'].tolist())

        chunks.append({
            'start_sentence_num': chunk['sentence_num'].iloc[0],
            'end_sentence_num': chunk['sentence_num'].iloc[-1],
            'text': chunk_text,
            'num_words': len(chunk_text.split(' '))
        })

    chunks_df = pd.DataFrame(chunks)
    return chunks_df.to_dict('records')


def parse_title_summary_results(results):
    out = []
    for e in results:
        e = e.replace('\n', '')
        if '|' in e:
            processed = {'title': e.split('|')[0],
                         'summary': e.split('|')[1][1:]
                         }
        elif ':' in e:
            processed = {'title': e.split(':')[0],
                         'summary': e.split(':')[1][1:]
                         }
        elif '-' in e:
            processed = {'title': e.split('-')[0],
                         'summary': e.split('-')[1][1:]
                         }
        else:
            processed = {'title': '',
                         'summary': e
                         }
        out.append(processed)
    return out


def summarize_stage_1(chunks_text):

    print(f'Start time: {datetime.now()}')

    # Prompt to get title and summary for each chunk
    map_prompt_template = """Firstly, give the following text an informative title. Then, on a new line, write a 75-100 word summary of the following text:
    {text}

    Return your answer in the following format:
    Title | Summary...
    e.g.
    Why Artificial Intelligence is Good | AI can make humans more productive by automating many repetitive processes.

    TITLE AND CONCISE SUMMARY:"""

    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])

    # Define the LLMs
    map_llm = OpenAI(temperature=0, model_name = 'text-davinci-003')
    map_llm_chain = LLMChain(llm = map_llm, prompt = map_prompt)
    map_llm_chain_input = [{'text': t} for t in chunks_text]
    # Run the input through the LLM chain (works in parallel)
    map_llm_chain_results = map_llm_chain.apply(map_llm_chain_input)

    stage_1_outputs = parse_title_summary_results([e['text'] for e in map_llm_chain_results])

    print(f'Stage 1 done time {datetime.now()}')

    return {
        'stage_1_outputs': stage_1_outputs
    }


def get_topics(title_similarity, num_topics = 8, bonus_constant = 0.25, min_size = 3):

    proximity_bonus_arr = np.zeros_like(title_similarity)
    for row in range(proximity_bonus_arr.shape[0]):
        for col in range(proximity_bonus_arr.shape[1]):
            if row == col:
                proximity_bonus_arr[row, col] = 0
            else:
                proximity_bonus_arr[row, col] = 1/(abs(row-col)) * bonus_constant

    title_similarity += proximity_bonus_arr

    title_nx_graph = nx.from_numpy_array(title_similarity)

    desired_num_topics = num_topics
    # Store the accepted partitionings
    topics_title_accepted = []

    resolution = 0.85
    resolution_step = 0.01
    iterations = 40

    # Find the resolution that gives the desired number of topics
    topics_title = []
    while len(topics_title) not in [desired_num_topics, desired_num_topics + 1, desired_num_topics + 2]:
        topics_title = community.louvain_communities(title_nx_graph, weight = 'weight', resolution = resolution)
        resolution += resolution_step
    topic_sizes = [len(c) for c in topics_title]
    sizes_sd = np.std(topic_sizes)
    modularity = community.modularity(title_nx_graph, topics_title, weight = 'weight', resolution = resolution)

    lowest_sd_iteration = 0
    # Set lowest sd to inf
    lowest_sd = float('inf')

    for i in range(iterations):
        topics_title = community.louvain_communities(title_nx_graph, weight = 'weight', resolution = resolution)
        modularity = community.modularity(title_nx_graph, topics_title, weight = 'weight', resolution = resolution)

        # Check SD
        topic_sizes = [len(c) for c in topics_title]
        sizes_sd = np.std(topic_sizes)

        topics_title_accepted.append(topics_title)

        if sizes_sd < lowest_sd and min(topic_sizes) >= min_size:
            lowest_sd_iteration = i
            lowest_sd = sizes_sd

    # Set the chosen partitioning to be the one with highest modularity
    topics_title = topics_title_accepted[lowest_sd_iteration]
    print(f'Best SD: {lowest_sd}, Best iteration: {lowest_sd_iteration}')

    topic_id_means = [sum(e)/len(e) for e in topics_title]
    # Arrange title_topics in order of topic_id_means
    topics_title = [list(c) for _, c in sorted(zip(topic_id_means, topics_title), key = lambda pair: pair[0])]
    # Create an array denoting which topic each chunk belongs to
    chunk_topics = [None] * title_similarity.shape[0]
    for i, c in enumerate(topics_title):
        for j in c:
            chunk_topics[j] = i

    return {
        'chunk_topics': chunk_topics,
        'topics': topics_title
    }


def summarize_stage_2(stage_1_outputs, topics, summary_num_words = 250):
    print(f'Stage 2 start time {datetime.now()}')

    # Prompt that passes in all the titles of a topic, and asks for an overall title of the topic
    title_prompt_template = """Write an informative title that combines each of the following groups of titles into one summary. Make sure that the titles capture as much information as possible,
    and are different from each other. Make sure that important data, such as references to specific documents or meetings, are captured:
    {text}

    Return your answer in a numbered list, with new line separating each title:
    1. Title 1
    2. Title 2
    3. Title 3
  
    TITLES:
    """

    map_prompt_template = """Write a 25-50 word summary of the following text:
    {text}

    CONCISE SUMMARY:"""

    combine_prompt_template = 'Write a ' + str(summary_num_words) + """-word summary of the following, removing irrelevant information. Finish your answer:
    {text}
    """ + str(summary_num_words) + """-WORD SUMMARY:"""

    title_prompt = PromptTemplate(template=title_prompt_template, input_variables=["text"])
    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
    combine_prompt = PromptTemplate(template=combine_prompt_template, input_variables=["text"])

    topics_data = []
    for c in topics:
        topic_data = {
            'summaries': [stage_1_outputs[chunk_id]['summary'] for chunk_id in c],
            'titles': [stage_1_outputs[chunk_id]['title'] for chunk_id in c]
        }
        topic_data['summaries_concat'] = ' '.join(topic_data['summaries'])
        topic_data['titles_concat'] = ', '.join(topic_data['titles'])
        topics_data.append(topic_data)

    # Get a list of each community's summaries (concatenated)
    topics_summary_concat = [c['summaries_concat'] for c in topics_data]
    topics_titles_concat = [c['titles_concat'] for c in topics_data]

    # Concat into one long string to do the topic title creation
    topics_titles_concat_all = ''''''
    for i, c in enumerate(topics_titles_concat):
        topics_titles_concat_all += f'''{i+1}. {c}
    '''

    # print('topics_titles_concat_all', topics_titles_concat_all)

    title_llm = OpenAI(temperature=0, model_name = 'text-davinci-003')
    title_llm_chain = LLMChain(llm = title_llm, prompt = title_prompt)
    title_llm_chain_input = [{'text': topics_titles_concat_all}]
    title_llm_chain_results = title_llm_chain.apply(title_llm_chain_input)


    # Split by new line
    titles = title_llm_chain_results[0]['text'].split('\n')
    # Remove any empty titles
    titles = [t for t in titles if t != '']
    # Remove spaces at start or end of each title
    titles = [t.strip() for t in titles]

    map_llm = OpenAI(temperature=0, model_name = 'text-davinci-003')
    reduce_llm = OpenAI(temperature=0, model_name = 'text-davinci-003', max_tokens = -1)

    # Run the map-reduce chain
    docs = [Document(page_content=t) for t in topics_summary_concat]
    chain = load_summarize_chain(chain_type="map_reduce", map_prompt = map_prompt, combine_prompt = combine_prompt, return_intermediate_steps = True,
                                 llm = map_llm, reduce_llm = reduce_llm)

    output = chain({"input_documents": docs}, return_only_outputs = True)
    summaries = output['intermediate_steps']
    stage_2_outputs = [{'title': t, 'summary': s} for t, s in zip(titles, summaries)]
    final_summary = output['output_text']

    # Return: stage_1_outputs (title and summary), stage_2_outputs (title and summary), final_summary, chunk_allocations
    out = {
        'stage_2_outputs': stage_2_outputs,
        'final_summary': final_summary
    }
    print(f'Stage 2 done time {datetime.now()}')

    return out

if __name__ == "__main__":
    main("../../cleaned_output", chunks_file="chunks.out", stage1_file="stage1.txt")
