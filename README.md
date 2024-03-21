[English/Português]
<br><br>

This repository contains the content taught in the first class of introduction to machine learning in the [DataEngineer Help](https://www.linkedin.com/company/dataengineerhelp/) community.
<br><br>
# Introduction to Machine Learning, Scikit-Learn, and KNN 
<br><br>

## What is Machine Learning?
Machine learning is a field of computer science that, unlike traditional computing where every action needs to be explicitly programmed, is focused on developing algorithms and mathematical/statistical models capable of "learning" the behavior of a phenomenon or identifying similarities from previous data, thus being able to make predictions about the phenomenon or diagnose it. 
<br><br>

## How do machines learn?
Could a fotographic memory save you in an exam? Well, without understanding the subject, it's likely that it wouldn't help you much, because your knowledge would be limited to exactly what you memorized - input of data. It is more effective to understand the general concepts and main ideas, so that we can abstract rather than trying to memorize everything that exists. Machines learn in a similar way. 
<br>

![image](https://github.com/lucas-mdsena/aula-intro-ml-knn/assets/93884007/bee99b33-693c-49ce-8f4d-6af481168f74)
<br>

Although it is written "This is not a pipe," we recognize that it is. Our brain is capable of abstracting a painting because it has been exposed to images of pipes from different angles for years, being able to generalize to recognize pipes of different colors, sizes, and shapes. To fit a machine learning model on a dataset is called training. Thus, it can search for key concepts in the abstracted data during training (pipes have a mouth, stem) to make decisions - to generalize. Just like our learning is related to what we are exposed to, the machine is too. The model will be as good as the data that was used for its training. If a facial recognition algorithm is trained using only people without glasses, when it sees a person with glasses, it will not say it's a face. It is not able to generalize well. We say that, in this case, the data has bias.
<br>

![image](https://github.com/lucas-mdsena/aula-intro-ml-knn/assets/93884007/28c4c2a3-d7fb-4d23-a643-2ab0020fd82d)
<br><br>

## Types of learning
For each type of situation, there is a more suitable learning approach that will involve different steps. The three ways a computer learns can be divided into **supervised learning**, **unsupervised learning**, and **reinforcement learning**.
<br><br>
**Supervised learning:** is so named because it is like learning from a teacher. The data used for learning already provides examples of the phenomenon to be learned, such as the price of a property, and other features that may or may not influence it, such as its size, location. In this example, the goal of the algorithm would be to predict the value of a new property based on the provided characteristics. 
<br><br>
**Unsupervised learning:** this time there is no teacher, as there is no phenomenon to be predicted or classified. The goal is to identify patterns in the provided data. For example, in a database of customers from a retailer, groups of customers more similar to each other could be identified based on their previous purchases, and grouped to target promotions more effectively. 
<br><br>
**Reinforcement learning:** similar to supervised, there are examples of the studied phenomenon in the database, but there is one more step. The output of the algorithm is always evaluated and compared to a possible expected result, such as a reward or penalty, in order to adjust the algorithm's parameters and obtain a better result. It's like training a dog. 
<br><br>

## Learning stages
**Data acquisition:** data can be available in structured form - xlxs files, csv, SQL database, semi-structured - json and xml, and unstructured - images and text.
<br><br>
**Exploration, cleaning, and data treatment:** again, a model will be as good as the data that was used for its training. This is the stage where you get to know the data and a good part of your time will be spent here, if not the most. In this process, the presence of outliers, missing values, normality of distributions should be checked. Questions should be asked and statistical tests performed. Correlation between predictor and predicted variables. Standardization or normalization of variable dimensions, encoding and choosing which ones can or cannot be eliminated from the database (feature engineering).
<br><br>
**Training the model on the data:** the technique is chosen according to the problem - there is no perfect technique. At this point, the model is adjusted to the dataset that was processed in the previous step. The relationship between the variables is mapped (inputs and outputs) to "learn" key concepts. The trained model represents the dataset it received.
<br><br>
**Testing and validation:** after training, the model receives data that was not seen during training. There are metrics for evaluating the quality of a model's output, such as accuracy, recall.
<br><br>
**Optimization:** the result that the model delivers in the test is not final. Each technique has parameters to be chosen for learning that will influence its result, such as the depth of a decision tree and the method of distance calculation in KNN.
<br><br>
**Deployment:** after fine-tuning the model, it can be put into production to make real-time predictions, classifications, or diagnoses. A model can be hosted in a cloud environment, implemented in software.
<br><br>

## Scikit-Learn
Scikit-Learn is an open-source Python library for predictive analysis, data preprocessing, and machine learning. It was developed using the NumPy and SciPy mathematical libraries. It offers a high-level approach to implementing models and machine learning and is widely used by Python users.
<br>
[Scikit-Learn](https://scikit-learn.org/stable/#)
<br><br><br>

# Classification with KNN
## Introduction
Imagine you are doing a blind taste test. You have all your years of knowledge about food and dishes, taste, texture, smell. So if you taste a peanut without seeing it, you will recognize it by taste, texture, and shape. In other words, you classified a food as a peanut based on its features. But what if it were a chestnut? It has a somewhat similar shape, it's also salty and crunchy. If we were to create groups, peanuts and chestnuts could certainly be grouped together based on these characteristics. We can say they are neighbors.
<br><br>
The nearest neighbors classification technique is capable of classifying an unlabeled observation according to the class of the nearest examples (neighbors) to it. It will work better as there is a clear distinction between groups and intra-group similarities. Otherwise, it may not be the best choice.
<br><br>

## KNN Algorithm
KNN (k-nearest neighbors) is the algorithm that implements nearest neighbors classification. 
<br>
It is trained with a database containing observations of n features (x) classified into various categories (y). A test set with different observations of the same features is submitted to the trained model, which, for each observation, will identify k records in the training data that are most similar to it. This test observation will be classified according to the class of the majority of the k neighbors.
<br><br>
**Advantages:**
<br>
> Algorithm of simple understanding
<br>
> The training stage is fast
<br>

**Disadvantages:**
<br>
> Very sensitive to outliers and missing data <br>
<br>

## Example:
<br><br>
![image](https://github.com/lucas-mdsena/aula-intro-ml-knn/assets/93884007/de55f4eb-0d72-4173-8fa7-e1d35a06cb46)
<br><br>
Since we have only two features, we can represent them in a Cartesian way:
<br><br>
![image](https://github.com/lucas-mdsena/aula-intro-ml-knn/assets/93884007/f83fca5c-6567-4955-b01b-906de38b1220)
<br><br>
We can notice that similar foods are closer to each other:
<br><br>
![image](https://github.com/lucas-mdsena/aula-intro-ml-knn/assets/93884007/8821697b-eba0-4a9d-a0c3-ef49d61cbf9c)
<br><br>
What happens if we try to insert a new food? We can use the nearest neighbors technique to determine how it will be classified.
<br><br>
![image](https://github.com/lucas-mdsena/aula-intro-ml-knn/assets/93884007/95fa2c28-9965-4980-b16e-d3cca0ca82d4)
<br><br>
To know which are the nearest neighbors of the tomato, it is necessary to calculate its distance to all the other neighbors, so only numerical features can be used in this algorithm. In the presence of categorical features, dummy encoding can be performed. The most traditional distance function is the Euclidean distance:
<br><br>
p and q are the observations for which their distance is being calculated:
<br><br>
![image](https://github.com/lucas-mdsena/aula-intro-ml-knn/assets/93884007/435fdab0-1cee-4762-a8a7-282a9748d691)
<br><br>
Let's say the tomato has a sweetness of 6 and crunchiness of 4, its distance to the other foods would be:
<br><br>
![image](https://github.com/lucas-mdsena/aula-intro-ml-knn/assets/93884007/c2d5e948-08f0-4f71-9ff1-a89eb9a4fcb4)
<br><br>
With the distances calculated, we must choose the number of neighbors we will compare, k. If k=1, then the tomato is classified as a fruit, since the smallest distance is 1.4. If we set k=3, it will choose the class that occurs most frequently. In this case, 2 fruits and 1 protein - classified as fruit again.
<br><br>
K is a hyperparameter of the algorithm and its choice will affect the accuracy of the final model. If we set k too high, it can be a way to shield against high variance in the data, but we may have a problem called overfitting - the model memorizes the training data and its ability to generalize will be very low.
<br><br>
On the other hand, if k is set to 1, we may have underfitting - the comparison factor of the model will be too low.
<br><br>
Commonly, the value of k is chosen as the square root of the number of observations in the training set. There is also the practice of testing with values between 3 and 10. In any case, the value of k should be explored, we will see this in the model optimization lesson.
<br><br>

## Data dimensionality
One care that should always be taken when dealing with KNN is regarding the dimensionality of the data. Since there is a calculation of distances to determine the classification, a feature with very large dimensions will cause an imbalance between the calculated values. To deal with this, we must bring all features to the same scale.
<br><br>
The most popular way to handle this in KNN is with min-max normalization. This process transforms the numbers to a scale between 0 and 1.
<br><br>
![image](https://github.com/lucas-mdsena/aula-intro-ml-knn/assets/93884007/d8182ebb-5dda-4beb-991e-f32a76695ec7)
<br><br>
Another approach is z-score standardization. No maximum and minimum values are defined, but it generates negative values, so this should be taken into account when performing certain statistical tests.
<br><br>
![image](https://github.com/lucas-mdsena/aula-intro-ml-knn/assets/93884007/126623d5-c88e-4ac5-afc2-1a80710e2386)
<br><br>

## Is KNN lazy?
Technically, there is no abstraction and generalization in the KNN algorithm, it only compares closest values. That's why it's called lazy learning. Since there is no abstraction, the training phase is much faster than other techniques, but predictions may be slower. It is said that this technique does not generate a proper model. Despite this, this technique should not be underestimated, after all, the best model is the one that delivers satisfactory results with low computational cost.
<br><br>

## References
LANTZ, Brett. Machine Learning With R.
<br><br>

***
***
Este repositório contém o conteúdo ministrado na primeira aula de introdução a machine learning, na comunidade [DataEngineer Help](https://www.linkedin.com/company/dataengineerhelp/)
<br><br>
# Introdução a Machine Learning, Scikit-Learn e KNN 
<br><br>

## O que é Machine Learning?
O aprendizado de máquina é uma área da computação que, diferentemente da computação tradicional em cada ação precisa ser devidamente programada, se destina ao desenvolvimento de algoritmos e modelos matemáticos/estatísticos capazes de “aprender” o comportamento de um fenômeno ou identificar similaridades a partir de dados prévios, assim sendo capaz de fazer previsões sobre o fenômeno ou diagnosticá-lo. 
<br><br>

## Como as máquinas aprendem?
Já desejou uma memória fotográfica quando estava estudando para uma prova? Bom, sem entender a matéria, é provável que isso não fosse lhe ajudar, pois seu conhecimento estaria limitado a exatamente ao que você memorizou – entrada dos dados. É mais eficaz entender os conceitos gerais e ideias principais, para sermos capazes de abstrair do que tentar decorar tudo que existe. Máquinas aprendem de um jeito parecido. 
<br>

![image](https://github.com/lucas-mdsena/aula-intro-ml-knn/assets/93884007/bee99b33-693c-49ce-8f4d-6af481168f74)
<br>

Apesar de estar escrito “Isso não é um cachimbo”, nós reconhecemos que é. Nosso cérebro é capaz de abstrair uma pintura, pois foi ajustado por anos com imagens de cachimbos nos mais diferentes ângulos, sendo capaz de generalizar para reconhecer carimbos de cores diferentes, tamanhos, formatos.
Ajustar um modelo de machine learning a um conjunto de dados é chamado de treinamento. Assim, ele pode buscar conceitos principais nos dados abstraídos durante o treino (cachimbos tem boca, cano) para tomar decisões – generalizar.
Assim como o nosso aprendizado está relacionado ao que somos expostos, a máquina também. O modelo será tão bom quanto os dados que foram usados para o seu treinamento. Se um algoritmo de reconhecimento facial for treinado usando apenas pessoas sem óculos, quando ele vir uma pessoa de óculos ele não dirá que é um rosto. Ele não é capaz de generalizar bem. Dizemos que, neste caso, os dados possuem viés. 
<br>

![image](https://github.com/lucas-mdsena/aula-intro-ml-knn/assets/93884007/28c4c2a3-d7fb-4d23-a643-2ab0020fd82d)
<br><br>

## Tipos de aprendizado
Para cada tipo de situação, há uma abordagem de aprendizado mais adequada que envolverá diferentes etapas. As três formas que um computador aprende podem ser divididas em **aprendizado supervisionado**, **não-supervisionado** e **aprendizado por reforço**. 
<br><br>
**Aprendizado supervisionado:** é assim chamado, pois é como aprender com um professor. Os dados usados para aprender já fornecem exemplos do fenômeno a ser aprendido, como o preço de um imóvel, e outras características que podem ou não ter influência sobre ele, como o seu tamanho, localidade. Neste exemplo, o objetivo do algoritmo seria prever o valor de um novo imóvel com base nas características informadas. 
<br><br>
**Aprendizado não supervisionado:** desta vez não há professor, pois não um fenômeno a ser previsto ou classificado. O objetivo é identificar padrões nos dados fornecidos. Por exemplo, em um banco de dados de clientes de uma rede varejista, poderiam ser identificados grupos de consumidores mais parecidos, conforme suas compras prévias, e agrupá-los para direcionar promoções de forma mais assertiva. 
<br><br>
**Aprendizado por reforço:** parecido com o supervisionado, há exemplos do fenômeno estudado na base de dados, mas há mais uma etapa. A saída do algoritmo é sempre avaliada e comparada com um possível resultado esperado, como uma recompensa ou penalidade, de forma a se ajustar os parâmetros do algoritmo e obter-se um resultado melhor. É como treinar um cachorro. 
<br><br>

## Etapas do aprendizado
**Obtenção de dados:** dados podem estar disponíveis na forma estruturada – arquivos xlxs, csv, banco de dados SQL, semiestruturada – json e xml, e não estruturada – imagens e texto.
<br><br>
**Exploração, limpeza e tratamento de dados:** novamente, um modelo será tão bom quanto os dados que foram usados para o seu treinamento. Esta é etapa em que se conhecem os dados e uma boa parte do seu tempo será nela, senão a maior. Neste processo devem ser verificados presença de outliers, missing values, normalidade de distribuições. Perguntas devem ser feitas e testes estatísticos realizados. Correlação entre variáveis preditoras e predita. Padronização ou normalização da dimensão de variáveis, encoding e escolha de quais podem ou não ser eliminadas da base de dados (feature engineering).
<br><br>
**Treino do modelo nos dados:** a técnica é escolhida de acordo com o problema – não há técnica perfeita. Neste momento, o modelo é ajustado ao conjunto de dados que foram tratados no passo anterior. A relação entre as variáveis é mapeada (entradas e saídas) para “aprender” conceitos chaves. O modelo treinado representa o conjunto de dados que recebeu.
<br><br>
**Teste e validação:** após o treinamento, o modelo recebe dados que não foram vistos durante o treino. Existem métricas para avaliação da qualidade da saída de um modelo, como precisão, recall.
<br><br>
**Otimização:** o resultado que o modelo entrega no teste não é final. Cada técnica possui parâmetros a serem escolhidos para o aprendizado que influenciarão em seu resultado, como a profundidade de uma árvore de decisões e o método de cálculo de distância no KNN.
<br><br>
**Deployment:** após o ajuste do modelo, ele pode ser colocado em produção para fazer previsões, classificações ou diagnósticos em tempo real. Um modelo pode ser hospedado em um ambiente cloud, implementado em um software.
<br><br>

## Scikit-Learn
Scikit-Learn é uma biblioteca open Source Python para análise preditiva, pré-processamento de dados e machine learning. Foi desenvolvida utilizando as bibliotecas matemáticas NumPy e SciPy. Ela oferece uma abordagem de alto nível para implementação de modelos e machine learning e é amplamente utilizada pelos usuários da linguagem.
<br>
[Scikit-Learn](https://scikit-learn.org/stable/#)
<br><br><br>

# Classificação com KNN
## Introdução
Imagine que vocês estão fazendo uma prova de comidas as cegas. Vocês têm toda a sua bagagem anos sobre alimentos e pratos, sabor, textura, cheiro. Então se você experimenta um amendoim sem vê-lo, você o reconhecerá pelo sabor, textura e formato. Em outras palavras, você classificou um alimento como amendoim pelas características - features. Mas e se fosse uma castanha? Formato um pouco parecido, também é salgada e crocante. Se fôssemos criar grupos, certamente amendoins e castanhas até poderiam ficar juntos seguindo essas características. Podemos dizer que elas são vizinhas.
<br><br>
A classificação por vizinhos próximos (nearest neighbors) é uma técnica capaz de classificar uma observação não rotulada conforme a classe de exemplos (vizinhos) mais próximos a ela. Ela funcionará melhor à medida que há uma distinção clara entre grupos e similaridades intragrupos. Caso contrário, pode não ser a melhor escolha.
<br><br>

## Algoritmo KNN
KNN (k-nearest neighbors) é o algoritmo que implementa a classificação por vizinhos próximos. 
<br>
Ele é treinado com uma base de dados contendo observações de n features (x) classificadas em diversas categorias (y). Uma base de teste com diferentes observações das mesmas features é submetida ao modelo treinado, que, para cada observação, identificará k registros nos dados de treino que são mais semelhantes a ela. Essa observação de teste será classificada conforme a classe da maioria dos k vizinhos.
<br><br>
**Vantagens:**
<br>
> Algoritmo de simples entendimento
<br>
> A etapa de treinamento é rápida
<br>

**Desvantagens:**
<br>
> Muito sensível a outliers e dados faltantes <br>
<br>

## Exemplo:
<br><br>
![image](https://github.com/lucas-mdsena/aula-intro-ml-knn/assets/93884007/de55f4eb-0d72-4173-8fa7-e1d35a06cb46)
<br><br>
Como temos apenas duas features, podemos representar de forma cartesiana:
<br><br>
![image](https://github.com/lucas-mdsena/aula-intro-ml-knn/assets/93884007/f83fca5c-6567-4955-b01b-906de38b1220)
<br><br>
Podemos notar que comidas parecidas estão mais próximas umas das outras:
<br><br>
![image](https://github.com/lucas-mdsena/aula-intro-ml-knn/assets/93884007/8821697b-eba0-4a9d-a0c3-ef49d61cbf9c)
<br><br>
O que acontece se tentarmos inserir um novo alimento? Podemos usar a técnica de vizinhos próximos para determinar como será classificado.
<br><br>
![image](https://github.com/lucas-mdsena/aula-intro-ml-knn/assets/93884007/95fa2c28-9965-4980-b16e-d3cca0ca82d4)
<br><br>
Para saber quais são os vizinhos mais próximos do tomate, é necessário calcular sua distância até todos os outros vizinhos, portanto apenas features numéricas podem ser usadas neste algoritmo. Na presença de features categóricas, pode ser realizada um dummização. A função de distância mais tradicional é a euclidiana:
<br><br>
p e q são as observações em que sua distância está sendo calculada:
<br><br>
![image](https://github.com/lucas-mdsena/aula-intro-ml-knn/assets/93884007/435fdab0-1cee-4762-a8a7-282a9748d691)
<br><br>
Digamos que o tomate tenha uma doçura 6 e crocância 4, sua distância para os outros alimentos ficaria:
<br><br>
![image](https://github.com/lucas-mdsena/aula-intro-ml-knn/assets/93884007/c2d5e948-08f0-4f71-9ff1-a89eb9a4fcb4)
<br><br>
Com as distâncias calculadas, devemos escolher a quantidade de vizinhos que vamos comparar, o k. Se k=1, então o tomate é classificado como fruta, pois a menor distância é 1,4. Se definirmos k=3, ele escolherá a classe que mais ocorre. No caso, 2 frutas e 1 proteína - classificado como fruta novamente.
<br><br>
K é um hiperparâmetro do algoritmo e sua escolha implicará na precisão do modelo final. Se definirmos k muito alto, pode ser uma forma de se blindar de uma alta variância nos dados, mas poderemos ter um problema chamado overfitting - o modelo decora os dados de treino e sua capacidade de generalizar ficará muito baixa.
<br><br>
Por outro lado, se k for 1, poderemos ter um underfitting – fator de comparação do modelo será muito baixo.
<br><br>
Comumente, o valor de k é escolhido como a raiz quadrada do número de observações no treino. Há também a prática de testar com valores entre 3 e 10. De toda forma, o valor de k deve ser explorado, veremos isso na parte de otimização de modelos.
<br><br>

## Dimensão dos dados
Um cuidado que sempre deve ser tomado ao lidar com KNN é quanto à dimensão dos dados. Como há um cálculo de distâncias para determinar a classificação, uma feature com dimensões muito grandes causará um desbalanço entre os valores calculados. Para lidar com isso, devemos trazer todas as features para a mesma escala.
<br><br>
A forma mais popular de se lidar no KNN é com a normalização min-max (min-max normalization). Este processo transforma os números para uma escala entre 0 e 1.
<br><br>
![image](https://github.com/lucas-mdsena/aula-intro-ml-knn/assets/93884007/d8182ebb-5dda-4beb-991e-f32a76695ec7)
<br><br>
Outra abordagem é padronização z-score (z-score standardization). Não valores máximo e mínimo definidos, mas gera valores negativos, portanto deve-se levar em consideração ao realizar certos testes estatísticos.
<br><br>
![image](https://github.com/lucas-mdsena/aula-intro-ml-knn/assets/93884007/126623d5-c88e-4ac5-afc2-1a80710e2386)
<br><br>

## KNN é preguiçoso?
Tecnicamente, não há abstração e generalização no algoritmo KNN, ele apenas compara valores mais próximos. Por isso é chamado de lazy learning. Como não há abstração, a fase de treino é bem mais rápida que outras técnicas, mas as previsões podem ser mais lentas. É dito que esta técnica não gera um modelo propriamente dito. Apesar disso, esta técnica não deve ser menosprezada, afinal o melhor modelo é aquele que entrega resultados satisfatórios e com um baixo custo computacional.
<br><br>

## Referências
LANTZ, Brett. Machine Learning With R.













