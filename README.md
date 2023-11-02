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













