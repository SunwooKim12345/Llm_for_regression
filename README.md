# Regression Using LLM

## Introduction
The goal of this project is to explore the use of the NT-5 model, a large language model (LLM), for performing regression tasks. Specifically, I aimed to predict a continuous value from numerical inputs using both linear and non-linear regression methods. The NT-5 model is a fine-tuned model of the T5 model, which is a transformer-based model. This investigation will help determine if the NT-5 model can effectively handle numerical regression tasks.

## AI Problem Statement
The AI problem addressed in this project is the prediction of a continuous numeric value from a set of numerical inputs. This involves both linear and non-linear regression, which require understanding and modeling complex relations between input features and the output. The model will learn the relations between features from the training dataset and test with the test dataset for measuring the accuracy of the model.

## Solution
To solve the above AI problem, I used the NT-5 model by fine-tuning it on regression tasks. I started with simple linear functions to establish baseline performance and then moved on to more complex non-linear functions. By fine-tuning the model, it could learn the relations of the features from the training data, leading to significantly better performance compared to models not fine-tuned for regression tasks.

## Research Questions
My question for this project is:
- “How well does the NT-5 model perform in predicting continuous values for regression tasks (both linear and non-linear)?”

## Experimental Setting
I used Friedman 1, 2, and 3 datasets from scikit-learn for non-linear regression, and a linear function from scikit-learn (`make_regression`) for linear regression. To evaluate my model, I used MAE (mean absolute error). I used a pre-trained NT-5 model and fine-tuned it. The model was trained with 2000 data points and divided into training, validation, and test sets in an 8:1:1 ratio. I compared the model performance with GPT-4o, GPT-4, GPT-3.5, and Llama-3. I used the OpenAI API for GPT-4o, GPT-4, and GPT-3.5 to get the predicted answer for the prompt, and I used Together.ai to get the predicted answer for the prompt. For these models, I tried few-shot learning for comparison by giving 50 training data points in the prompt and one test data point in the prompt, then asking the model to predict the exact value.

## Result
From the results, I found that most tasks performed well for NT-5 since it was the fine-tuned model. These results illustrate that the NT-5 model could predict continuous values quite accurately in both linear and non-linear contexts.
![image](https://github.com/SunwooKim12345/Llm_for_regression/assets/129953673/4c4e7d03-e4d9-458d-8476-70e914622a0d)


## Main Takeaway
From this project, I learned that a large language model could predict numerical values for both linear and non-linear functions effectively. Additionally, fine-tuning the model significantly improves its performance. Furthermore, the success of the NT-5 model in this project demonstrates its potential applicability in predicting other complex real-world problems.

## Resources
- [NT-5 Model on Hugging Face](https://huggingface.co/nielsr/nt5-small-rc1)
- [LangChain OpenAI Integration](https://python.langchain.com/v0.2/docs/integrations/chat/openai/)
- [NT-5 Model Research Paper](https://arxiv.org/pdf/2404.07544.pdf)
