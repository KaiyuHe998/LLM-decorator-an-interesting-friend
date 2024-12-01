# NLP-decorator-an-interesting-friend
A project that use explainable LLM decorator to control the behavior of LLMs.
A demo video can be seen at [https://youtu.be/oHH1w6hXTeI](https://youtu.be/oHH1w6hXTeI)

# Overall Goal
- 1, Enhance Conversational Experience: Our primary goal is to transform the interaction with chatbots from a mere tool into a more engaging and human-like conversation. We aim to recapture the fun and enjoyment that was present when we first interacted with GPT. By making the chatbot's responses more lively and interesting, we want users to experience the same level of engagement and amusement that was once common, such as the witty comebacks and entertaining exchanges that occurred, especially with early versions like Bing's chatbot.
- 2, Simplify Usage and Interaction: Another key objective is to streamline and simplify the use of GPT, reducing the need for extensive prompt engineering. Currently, achieving a similar engaging interaction with existing chatbots often requires prewriting numerous prompts or finding specific instructions to bypass internal limitations. We aim to enable GPT to meet user expectations and deliver a satisfying conversational experience without the need for extensive pre-prepared prompts, thereby making the interaction more natural and effortless.

# Scope
- The project will focus on developing a limited prototype that demonstrates the extraction and control of vectors associated with a few selected semantic concepts (e.g., truthfulness, power-seeking, and emotion). The prototype will build a small database of vectors corresponding to these concepts, which can be dynamically used to guide the model's output during text generation. This prototype will focus on NLP tasks like question-answering and text generation to showcase how controlling internal representations can lead to different AI behavior.

# Data sources
- We plan to utilize a subset of the data from the paper Representation Engineering: A Top-Down Approach to AI Transparency by Zou et al. (2023) as a foundational source.
- This dataset focuses on representation engineering techniques that enhance transparency in AI systems by analyzing cognitive phenomena like honesty, power-seeking, and utility estimation. To align with our project's specific objectives, we will extract relevant portions of this dataset and manually augment it with newly created data(our contribution). 
- Our primary goal in augmenting the dataset is to design a dataset that includes multiple emotional factors, enabling the model to learn from these distinct emotional dimensions.
- If the dataset is effectively constructed, the machine can simulate a larger range of emotional responses and switch dynamically between different emotional states based on a hard-coded emotion control parameter.
We could further introduce a role-play mechanism in which the model emulates characters with specific emotional profiles, such as those from movies.
##
### 
- We developed our dataset using Shaver's [2] Emotional Model, categorizing emotions accordingly. We crafted and provided multiple rephrased variations for each phrase. Our dataset is distributed across 27 + neutral emotions, and we will train our model based on an augmented dataset(with emoji).
###
- We developed our dataset by drawing inspiration from dialogues in Christopher Nolan's The Dark Knight Trilogy [3], including Batman Begins (2005), The Dark Knight (2008) (Mainly), and The Dark Knight Rises (2012). Sentences attributed to Batman emphasize themes of justice, morality, and resilience, while Joker's lines convey chaos, unpredictability, and dark humor. We included its aim to enable the model to seamlessly switch between the Joker's and Batman's tones as an additional feature. This feature allows for dynamic tone modulation, offering nuanced control over stylistic outputs. Each sentence was rephrased or crafted to maintain the characters' essence, ensuring the dataset supports flexible and context-sensitive tone adaptation during training.

# Project Structure

- **Notebooks**
  - `Pipeline_example_experiment_evaluation.ipynb`: Demonstrates the entire process, including training and evaluating our EMO-chat-bot. It also includes usage examples. All steps of the project, from start to finish, are directly visible in the output of each code chunk.
  - `Visualization.ipynb`: Contains all the code for visualizations. The output figures are directly stored in the `Figures` directory.

- **data_aug**: Contains the data used to train our EMO-chat-bot.

- **data_eval**: Consists of the sampled questions used to evaluate our EMO-chat-bot.

  - Below are some sample evaluation results generated in response to sample questions:
  
    | Question                          | Emotion | Coherence Score | Coherence Analysis           | Engagement Score | Engagement Analysis | Empathy Score | Empathy Analysis |
    |-----------------------------------|---------|-----------------|------------------------------|------------------|---------------------|---------------|------------------|
    | Can I borrow some money from you? | Honest  | Yes             | Too long; omitted            | No               | Too long; omitted   | No            | Too long; omitted|
    | Peter and Mary broke up.          | Sadness | Tie             | Too long; omitted            | Yes              | Too long; omitted   | Yes           | Too long; omitted|
    | How are you feeling today?        | Neutral | Yes             | Too long; omitted            | Yes              | Too long; omitted   | Yes           | Too long; omitted|
    | What is your favorite color?      | Curious | No              | Too long; omitted            | Yes              | Too long; omitted   | No            | Too long; omitted|

- **Figures**: Stores all visualizations used in the report. The visualizations directly come from the csv file in `data_eval`.
  - [View Emotion Score PDF](Figures/emotion_score.pdf)
  - [View Emotion Usage PDF](Figures/emotion_usage.pdf)
  - [View Overall Result PDF](Figures/overall_result.pdf)

- **reference_project**: Contains the project code from RepE[1]. We utilized one of its data preprocessing functions.

- **scripts**: Contains our project code.
  - `scripts/hooked_transformer.py`: Implements our enhanced transformer model that supports more fine-grained adapters. It can select any `torch.nn.Module` and add an adapter (hook) to it. The hook can train PCA component directions on any module and decorate the hidden state as needed when forward is called.
  - `scripts/data_preprocessing.py`: Handles data preprocessing and includes helper functions.


# Team members
- Kaiyu He (kxh230002)
  - 1, Implement the adaptor method in the paper ”Representation Engineering”.
- Xiaokai Rong (xxr230000 )
  - 2. Dataset augmenting.
- Jia Li (jxl220096)
  - 3. Semantic Classification.

#Reference:
- [1]Representation Engineering: A Top-Down Approach to AI Transparency 2023, arXiv, https://arxiv.org/abs/2310.01405. Access Date: Oct 07, 2024
- [2]P. R. Shaver, J. C. Schwartz, D. Kirson, and C. O’Connor, “Emotion knowledge: further exploration of a prototype approach.” Journal of Personality and Social Psychology, vol. 52 6, pp. 1061–86, 1987.
- [3]Nolan, C., Nolan, J., & Goyer, D. S. (2008). The Dark Knight: Screenplay. Warner Bros., https://www.nolanfans.com/library/pdf/thedarkknight-screenplay.pdf Access Date: Nov 20, 2024

