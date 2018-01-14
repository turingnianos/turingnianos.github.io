---
layout: post
tags: redes-neuronales deep-learning inteligencia-artificial
date: 2018-01-14
thumbnail: /images/dnn.png
title: "Aprendizaje profundo: una introducción"
comments: true
mathjax: true
published: true
---

Este post tiene como finalidad descubrir el intrigante mundo de las redes neuronales: una técnica de IA que en la última década ha generado resultados increíbles. Aprenderás por fin qué es eso del Deep Learning, como funciona una red neuronal y de qué modo se puede aplicar a diferentes problemas y situaciones del mundo real: detectar objetos en imágenes, generar música, traducir automáticamente de un idioma o vencer al ser humano en juegos como el Go o el Dota 2. Tranquilo, no hace falta ningún concepto previo de matemáticas o informática para entender el post.

<!--more-->

### Intersecciones: matemáticas, informática y biología
Si hay algo que me apasiona del conocimiento humano es cuando conceptos aparentemente inconexos se encuentran de repente. Si uno lo piensa bien, los mayores descubrimientos del ser humano resultan de pensar las cosas de un modo diferente. Para razonar usamos conceptos, de modo que esto lo podemos ver de otro modo: las más brillantes ideas surgen de asociar conceptos de forma original, diferente.

La [Inteligencia Artificial](https://turingnianos.github.io/2018/01/04/ia-que-es.html) se construye fundamentalmente a partir de conexiones de este tipo. Como el objetivo es lograr realizar de forma automática tareas propias del hombre y es el hombre el que trata de hacerlo, inevitablemente (para bien o para mal) aplica lo que percibe del mundo que le rodea. Es así como surgen por ejemplo los algoritmos genéticos (una técnica de aprendizaje automático): aplicando a la informática la idea de tener una población en la cual los que mejor se adaptan al entorno sobreviven y se reproducen dando lugar a variaciones en la especie. Como veis, esta técnica (que ya explicaremos en otro post) se basa en lo que en biología son dos de los mecanismos básicos de la evolución (selección natural y mutación).

Podríamos poner ejemplos de estas intersecciones con episodios como la manzana de Newton o el propio Eureka de Arquímedes, pero vamos a optar por otra más reciente: las redes neuronales artificiales.

### Neuronas y redes neuronales:

Como algunos recordaréis del colegio, el cerebro cuenta con unas células específicas del sistema nervioso (aunque también están en otras partes como la médula espinal y en el sistema digestivo). Estas células reciben el nombre de **neuronas** y se encargan de transmitir impulsos eléctricos a otras células, ya sean estas también nerviosas o de otro tipo (musculares por ejemplo). No sé si en su día viste *Érase una vez el cuerpo humano*, pero ahí se veía de una forma muy simple cómo unos muñecos correteaban de un lado para otro con las órdenes que tenían que dar. Si tienes chavales pequeños lo recomiendo, tiene su gracia y aprenden bastante (aunque sea del año de la tana). Estos muñecos son los impulsos eléctricos que transmiten la información por medio de las neuronas.

<center>
  <br>
  {% include youtube_player.html id="iGsC-Yam6PM" %}
  <br>
  <p>
    <b>Érase una vez el cuerpo humano:</b> <i>las neuronas. Dibujos animados en los que explican las neuronas para niños.</i>
  </p>
</center>


Si bien lo anterior nos podría dar una idea algo general de lo que es una neurona, veamos con más detalle cómo funciona. Esto nos va a servir para poder entender mejor lo siguiente.
            
<div>
  <br>
  <figure>
    <center>
     <img src="/images/neurona.png" height="100%" width="100%" alt="Neurona" />
     <figcaption><b>Figura 1:</b> <i>Partes principales de una neurona.</i></figcaption>
    </center>
  </figure>
  <br>  
</div>

Como podéis observar en la imagen, una neurona se compone de varias partes. Es importante entender las neuronas como vías por las cuales se transmite información (en forma de impulsos eléctricos). A nosotros nos interesa en particular lo siguiente:
* Las neuronas tienen un **cuerpo celular** llamado soma que contiene el núcleo.
* Las ramificaciones del cuerpo celular se llaman **dentritas** y se "conectan" con terminales nerviosas de otras neuronas.
* Hay también una extensión separada que suele ser más larga que recibe el nombre de **axón** y de la cual surgen las **terminales nerviosas** de la neurona.


Por tanto, para recapitular podemos decir que las neuronas tienen un cuerpo celular que tiene por un lado dentritas con las cuales otras neuronas se conectan con ella y terminales por las cuales ella se conecta a otras neuronas. Si ayuda, puedes ver las dentritas como diferentes entradas de USB que conectan con el cuerpo de la neurona y las terminales como una ramificación de diferentes USBs provenientes de un cable (axón) que sale desde el cuerpo.

De este modo, la información se transmite mediante un impulso eléctrico desde una neurona a otras. Dicho impulso se transmitirá mediante las terminales de la neurona a las dentritas de otras a las que esté "conectada". Tras llegar a las dentritas el cuerpo celular valorará si la señal es relevante y, de serlo, emitirá una señal por medio del axón para tratar de trasmitirla a otras neuronas por medio de sus terminales.

Muy bien, esto es muy bonito, pero ¿cómo interpreta el cuerpo celular una señal? ¿Cómo es eso de que decide si es relevante? Pues bien, para entender esto es clave saber que debido a la cantidad de información de diferente origen y formato que el ser humano tiene que manejar **las neuronas están especializadas en mayor o menor medida**. De este modo, se dice que dos neuronas pueden reaccionar de un modo muy diferente ante una misma señal eléctrica, y esto se debe principalmente a:
1. Cómo sea esa señal (de mayor o menor intensidad).
2. Cómo sea esa neurona (si está más o menos relacionada con señales de ese tipo: no es lo mismo que sea una neurona especializada en señales recibidas a partir de la vista que si está especializada en información olfativa).
3. Cómo de fuerte sea la conexión con la neurona por la cual le ha venido el impulso.

Como es de suponer, una  **red neuronal** es un sistema en el cual múltiples neuronas se encuentran interconectadas. El cerebro es por tanto una gran red neuronal.

Perfecto, hasta aquí la parte de biología. Ahora toca nuestro momento de intersección. Vamos intentar describir este modelo de la naturaleza de forma matemática. No, no te asustes, sigue leyendo. Ya verás como lo entiendes y te sorprende.

### Neuronas artificiales:

Pensemos un poco a cerca de lo anterior: *dos neuronas pueden reaccionar de un modo muy diferente ante una misma señal eléctrica*. Ok, entonces podemos decir que (de forma algo simplista) una neurona se conecta de diferente modo con otras. Es decir, que no es como un simple interruptor: no es que esté conectada *o* no lo esté ($1$ ó $0$), si no más bien que la conexión de una neurona con otra puede ir *desde* no conectada en absoluto *hasta* completamente conectada (desde $0$ hasta $1$).


#### Entradas, salidas y bias:
Perfecto, pues ahora tratemos de representar esta idea de neurona de una forma más esquemática, más conceptual. Para eso hace falta que introduzcamos un poquito de terminología en los 3 factores que planteamos antes:
1. La señal es un **input o entrada**, que se suele representar como $x$. Si hay $3$ señales de entrada (una por cada dentrita) tendríamos $x_{1}$, $x_{2}$, $x_{3}$.
2. Cada neurona tiene cierta tendencia o inclinación diferente a activarse según cómo sea. Esta parcialidad se denomina **bias** (del inglés), y se representa como $b$. Por tanto, si tenemos $2$ neuronas, cada una tendrá un bias asociado, $b_{1}$ el primero y $b_{2}$ el segundo.
3. Finalmente, cada conexión entre un input (cada señal) con la neurona tiene un valor llamado peso que puede ir desde $0$ hasta $1$ (como una probabilidad). Este valor se suele denominar **weight** (del inglés) y se representa como $w$. Es decir, cada input se relaciona con una neurona mediante un peso, de forma que no todos los inputs tienen la misma importancia para que la neurona se active o no.

La forma más minimalista de representar una neurona es el diagrama de abajo. En este diagrama sólo tenemos impulsos (azules) conectados con el núcleo (verde) para luego dar paso a una **salida o output** (rojo), representada con $y$ y que en este caso simplemente sería si se ha activado o no la neurona según los impulsos sean lo suficientemente influyentes frente al bias (si hubiera $3$ posibles outputs en vez de sólo 1 sería entonces $y_1$, $y_2$, $y_3$). Ahora vemos cómo se decide cuándo la neurona se enciende y demás.

<div>
  <br>
  <figure>
    <center>
     <img src="/images/nn0.png" height="100%" width="100%" alt="Diagrama0" />
     <figcaption><b>Diagrama 1:</b> <i>Esquema simple de una neurona.</i></figcaption>
    </center>
  </figure>
  <br>  
</div>

Como se puede observar, a cada columna (aunque esté formada sólo por un elemento) se le llama capa, de modo que se tiene en primer lugar la **capa de entrada**, después una **capa oculta** y por última la **capa de salida**. La capa oculta recibe este nombre porque no es algo que se pueda observar directamente como pueda ser el input o el output. Etiquetemos un poco nuestro diagrama en base a estas definiciones, obteniendo el diagrama 2.

<div>
  <br>
  <figure>
    <center>
     <img src="/images/nn1.png" height="100%" width="100%" alt="Diagrama1" />
     <figcaption><b>Diagrama 2:</b> <i>Esquema simple de una neurona con etiquetas para los inputs ($x$), bias ($b$) y output ($y$).</i></figcaption>
    </center>
  </figure>
  <br>  
</div>

#### Pesos y activación:
Ya sabemos cómo representar de forma esquemática y simbólica las entradas (inputs), la parcialidad (bias) y la salida (output) de una neurona, pero aún nos quedan dos aspectos clave: los pesos (weights) y cómo saber cuándo se activa la neurona. Vamos con ello.

En primer lugar, cómo ya vimos antes, los pesos son valores $w$ de entre $0$ y $1$ que indican lo importante que es la conexión de un input concreto con una neurona determinada, pero vamos a generalizar esto un poco más: un peso va a indicar directamente la relevancia de la conexión entre un elemento de una capa y otro elemento de la siguiente capa. Podemos de hecho ver las flechas que conectan elementos como **operaciones**, de modo que nuestra neurona opera con las entradas para decidir si se activa o no. Veamos estas operaciones.

En primer lugar, puesto que los weights son como probabilidades tiene sentido pensar que afectarán directamente a cada input. Dicho de otro modo: 
* Si tenemos un input con un valor $x$ y un peso que lo relaciona con otro elemento de valor $w = 1$ tiene sentido pensar que el valor de ese input para el otro elemento que lo recibe será de $1 * x = x$, es decir que será directamente $x$. 
* El otro extremo será cuando el peso es de $w = 0$, por lo cual tiene sentido pensar que será directamente $0 * x = 0$, de modo que $x$ no influirá en absoluto para el elemento.

De este modo, como vimos que los pesos pueden ir de $0$ a $1$ se puede ver cómo **los pesos son directamente proporionales a los inputs**, es decir que siempre se multiplicará para saber su importancia de cara a la conexión considerada: $w * x$.

Por otro lado, ¿cómo decide nuestra neurona cuándo activarse y cuándo no? Pues bien, para eso tiene sentido que deba primero tomar todos los inputs con sus pesos y comparar todo ello con lo dada que es a activarse la neurona, es decir, con su bias. Más formalmente, si tenemos $3$ inputs ($x_1$, $x_2$, $x_3$) con un peso asociado cada uno de conexión con la neurona ($w_1$, $w_2$, $w_3$ respectivamente) y nuestra neurona tiene un bias $b$ habrá que comparar $w_1 * x_1 + w_2 * x_2 + w_3 * x_3$ con $b$. Por esto que acabamos de ver **el bias suele ser negativo**, puesto que es como la inercia de la neurona, lo que hace que la neurona se oponga a activarse. De este modo podríamos directamente sumar a nuestro resultado el bias, ya que al ser negativo lo que va a hacer es restar:

$$w_1 * x_1 + w_2 * x_2 + w_3 * x_3 + b$$

Esta expresión significa: *"la activación de una neurona depende de todas sus entradas ($x$) con sus respectivas importancias ($w$) junto con lo propensa que sea a activarse de por sí ($b$)"*. Para los que les guste poner las cosas más compactas, en matemáticas se usa una notación con un símbolo denominado sumatorio: si en vez de referirnos siempre a cada entrada una por una ($x_1$, $x_2$, ...) nos referimos siempre a una entrada genérica $x_i$, con ir dando valores a $i$ obtenemos cada entrada particular, y lo mismo con los pesos. De este modo la expresión pasa a ser:

$$\sum_{i = 1}^{3}(w_i*x_i) + b$$ 

que significa *"la suma desde $i = 1$ hasta $i = 3$ de el peso $w_i$ por la entrada $x_i$, y todo ello sumado al bias $b$"*. Es decir, que se hace lo mismo para los $3$ valores de $i$. 

Ahora sólo nos queda encontrar algún tipo de regla que nos indique cuándo el resultado debe hacer que la neurona se active. Esto se hace por medio de una **función de activación**, que es una fórmula en la cual nosotros podemos introducir la expresión anterior y obtener una salida. Esta función se suele llamar $\sigma$ (la letra griega sigma). La salida puede ser tan sencilla como "me activo" ($0$) o "no me activo" o, al igual que ocurría con los pesos, más compleja admitiendo también "me activo un poco" ($0.3$), esto depende de la función que escojamos. A continuación exploraremos un poco cómo funciona esto de las funciones de activación y en qué consiste. Todo lo anterior está representado el diagrama 3:

<div>
  <br>
  <figure>
    <center>
     <img src="/images/nn2.png" height="100%" width="100%" alt="Diagrama2" />
     <figcaption><b>Diagrama 3:</b> <i>Esquema de la neurona con etiquetas para los inputs ($x_i$), bias ($b$) y output ($y$), incluyendo además los pesos ($w_i$) y la función de activación ($\sigma$).</i></figcaption>
    </center>
  </figure>
  <br>  
</div>

#### Función escalón:
Las **funciones** son relaciones matemáticas, lo que hacen es asociar un elemento de un tipo con otro (que puede ser de otro tipo). En nuestro caso según el valor que reciban, devuelven un valor diferente. Vamos a empezar analizando la función de activación más simple: la función escalón. Esta función es muy intuitiva, pues consiste básicamente en lo siguiente:
* Si el valor que recibe es menor o igual que un valor concreto devuelve el valor $0$ (es decir, que no se activa).
* Si el valor que recibe supera a dicho valor concreto devuelve el $1$ (es decir, se activa).
Este valor concreto se suele denominar **umbral** (*threshold* en inglés). Vamos a llamar al umbral $u$ y al valor que recibe la función $z$ (se suele usar $x$ para este último, pero no quiero que lo confundamos con los inputs). De este modo, matemáticamente esta función se expresaría del siguiente modo:

$$\sigma (z) = \left\{\begin{matrix}
0, & si  & z \leq u.\\ 
1, & si & z > u.
\end{matrix}\right.$$

Vamos a ver ya que estamos cómo es la función si se pinta (dando valores a $z$) y se toma como umbral $u = 0$, es decir que se activa si el valor es positivo ($z > 0$):

<div>
  <br>
  <figure>
    <center>
     <img src="/images/step.png" height="100%" width="100%" alt="Escalón" />
     <figcaption><b>Gráfico 1:</b> <i>Función escalón para un umbral de $0$.</i></figcaption>
    </center>
  </figure>
  <br>  
</div>

Esta función sólo admite dos salidas: $1$ y $0$. Como se puede observar, tiene sentido que se denomine función escalón. 

#### Función lineal:

Alguien podría plantearse en usar una función lineal como función de activación. De forma muy resumida, se dice que algo es lineal cuando se puede expresar como sumas y productos. Por tanto, una función lineal sería aquella en la que la salida es proporcional al valor recibido: 

$$\sigma(z) = m * z + n$$

Si uno lo piensa un poco esto es lo que se hace previamente a aplicar una función de activación: la expresión anterior es una expresión lineal ya que multiplica pesos y entradas sumándolos entre sí junto con un bias. Pero el principal problema de esto es que si queremos aplicar una función de activación lineal vamos a obtener un valor que no está acotado como en el caso de la función escalón. Es decir, nos interesa obtener siempre un valor que esté dentro de un rango: entre $0$ y $1$ por ejemplo. Por otro lado, las funciones lineales son rectas, así que tienen una variación constante (la pendiente). La variación, tal y como veremos más adelante en otro post, es fundamental para que las neuronas puedan aprender. Por tanto, no tiene sentido aplicar una función de activación lineal.

#### Función sigmoide: no linealidad

La función sigmoide es una de las más empleadas en aprendizaje profundo, de hecho he de decir que la función escalón en la práctica no se usa en este área si no que es más bien el primer modelo que se introdujo y que marcó el inicio de los modelos neuronales artificiales. Esto se debe principalmente a dos aspectos propios de la función sigmoide y otras frente a una función escalón o una lineal:
1. La función sigmoide introduce también las respuestas "me activo un poco" que comentábamos antes, de modo que coge cada valor posible que pueda recibir y devuelve un número *entre* $0$ y $1$ en función del valor recibido.
2. La función sigmoide no es lineal, de modo que introduce una **no linealidad** en el modelo.

Vamos a entender un poco estos aspectos, para ello veamos cómo es la función sigmoide y por qué tiene ese nombre tan raro. Sólo para los curiosos, la fórmula es la siguiente:

$$\sigma(z) = \frac{1}{1 + e^{x}}$$

Y la gráfica, que es lo que nos importa ahora, es esta:

<div>
  <br>
  <figure>
    <center>
     <img src="/images/sigmoid.png" height="100%" width="100%" alt="Sigmoide" />
     <figcaption><b>Gráfico 2:</b> <i>Función sigmoide.</i></figcaption>
    </center>
  </figure>
  <br>  
</div>

Como puedes ver, tiene forma de *s*, de ahí el nombre de la función, por la letra griega sigma (s). Lo interesante de esta función es que si uno la compara con la escalón la función avanza progresivamente de $0$ a $1$. Esta función tiene una variación que no es constante de modo que no presenta el problema de las funciones lineales. Esto significa que la función no es lineal.

Este asunto de **la no linealidad es fundamental para el aprendizaje profundo**, porque es lo que hace que algo pase de ser simplemente una combinación mediante sumas y productos a tener un comportamiento más complejo y menos predecible que hace que se puedan captar estructuras más complejas en la información.

#### Otras funciones de activación:

Además de la sigmoide, exiten otras funciones de activación no lineales que se usan bastante en la actualidad. Entre estas se encuentran:
- **Función tanh**: es una variación de la función sigmoide que tiene otra escala. Va desde $-1$ hasta $1$. Su fórmula es:

$$\sigma(z) = \frac{1}{1 + e^{-2x}} - 1$$

Y la gráfica:

<div>
  <br>
  <figure>
    <center>
     <img src="/images/tanh.png" height="100%" width="100%" alt="tanh" />
     <figcaption><b>Gráfico 3:</b> <i>Función tanh.</i></figcaption>
    </center>
  </figure>
  <br>  
</div>

- **Función ReLU**: cuenta con un umbral en $0$ de modo que si el valor es mayor que $0$, devuelve directamente el valor y en caso contrario devuelve $0$. Se suele aproximar con la función $\sigma(z) = ln(1 + e^{z})$. La fórmula es:

$$\sigma(z) = \max{(0, z)} = \left\{\begin{matrix}
0, & si  & z \leq 0.\\ 
z, & si & z > 0.
\end{matrix}\right.$$

$$\sigma(z) \approx ln(1 + e^{z})$$

Y su gráfica:

<div>
  <br>
  <figure>
    <center>
     <img src="/images/relu.jpeg" height="100%" width="100%" alt="ReLU" />
     <figcaption><b>Gráfico 4:</b> <i>Función ReLU.</i></figcaption>
    </center>
  </figure>
  <br>  
</div>

### Perceptrón simple:

Si recapitulamos un poco, podemos resumir que lo que hacemos por tanto es tomar cada entrada ($x_i$) con su peso asociado ($w_i$) y sumarlas junto con un bias ($b$). Tras resolver esta expresión pasamos el valor obtenido a una función de activación ($\sigma$) que lo toma y según cómo sea produce una salida ($y$):

$$y = \sigma(w_1 * x_1 + w_2 * x_2 + w_3 * x_3 + b) = \sigma(sum_{i = 1}^{3}(w_i * x_i) + b)$$

Este modelo se denomina **perceptrón simple** y fue introducido en 1957 por Frank Rosenblatt. A continuación se muestra un diagrama que contiene de forma resumida todo lo visto anteriormente.

<div>
  <br>
  <figure>
    <center>
     <img src="/images/nn3.png" height="100%" width="100%" alt="Diagrama3" />
     <figcaption><b>Diagrama 4:</b> <i>Esquema completo de una neurona artificial (perceptrón simple).</i></figcaption>
    </center>
  </figure>
  <br>  
</div>

### Redes neuronales artificiales:

Ahora que ya sabemos cómo funciona una neurona artificial podemos pasar a analizar redes de dichas neuronas, es decir, lo que se conoce como **redes neuronales artificiales**. Un aviso: he decidido adoptar la notación inglesa dado que es la de mayor uso en el área, de modo que a partir de ahora cuando me refiera a redes neuronales artificiales usaré las siglas **ANNs** (de Artificial Neural Networks), no RNA. Dicho esto, cabe añadir que a partir de ahora cada nodo (cada circulito) de los diagramas que aparezcan será una neurona, con su propio bias y su propia función de activación, que estará conectada a otras neuronas. Esto recuerda a la estructura del cerebro: una gran red neuronal.

Para entender las ANNs, debemos considerar lo siguiente:
- Las neuronas se presentan en capas y cada capa realiza una transformación concreta con la información que recibe. Recuerda que la primera capa y la última son casos especiales: la primera es la capa de entrada de información (no tiene bias ni pesos ya que no hay capas anteriores) y la última es la de salida (no tiene pesos que apunten a neuronas de la siguiente capa puesto que es la última).
- Las neuronas se especializan dentro de cada capa en diferentes aspectos según cuáles sean sus conexiones con la información que reciben (weights) y su tendencia (biases).
- Las transformaciones de la información junto con la especialización de cada neurona y una función no lineal dan lugar a una red que capta estructuras complejas de información (como puedan ser una imagen o un texto).
- Cuando una red neuronal tiene más de una capa oculta se considera una **red neuronal profunda** (**DNN**, de Deep Neural Network). Por tanto el aprendizaje profundo (Deep Learning) consiste en aplicar red neuronales profundas para lograr aprender a realizar tareas.

A continuación se muestra un diagrama con la estructura de una DNN con $2$ capas ocultas ($h1$ y $h2$):

<div>
  <br>
  <figure>
    <center>
     <img src="/images/dnn.png" height="100%" width="100%" alt="Diagrama4" />
     <figcaption><b>Diagrama 5:</b> <i>Arquitectura de una DNN (red neuronal profunda) con $2$ capas ocultas.</i></figcaption>
    </center>
  </figure>
  <br>  
</div>

### Aprendizaje profundo (Deep Learning): conclusión

El aprendizaje profundo como hemos visto es una rama de la Inteligencia Artificial que consiste en afrontar diferentes problemas por medio de redes neuronales profundas (DNNs). Estos modelos se basan en conceptos de la segunda mitad del siglo pasado, pero es en la última década cuando se han producido los mayores avances debido a dos aspectos fundamentales:
1. **Big Data**: actualmente se dispone de gran cantidad de datos y los modelos neuronales necesitan muchos datos para poder aprender correctamente estructuras complejas. Esto se debe a que el aprendizaje profundo se basa en la iteración sobre los datos para lograr aprender correctamente a predecir, de modo que para que dicha iteración sea eficaz y suficiente debe haber una cantidad de datos considerable.
2. **Capacidad computacional**: se han producido grandes avances en el desarrollo del hardware empleado para entrenar redes neuronales, concretamente las unidades de procesamiento gráfico (GPUs).

Por otro lado, estos modelos logran aprender con bastante eficacia debido en gran parte a la no linealidad que se introduce y a la especialización de sus elementos (las neuronas) en diferentes aspectos de la información tratada. Para ello como veremos en otro post, lo que se entrenan son los weights y los biases para lograr una estructura que sepa extraer correctamente las propiedades de la información.

Tal y como se explicará mas adelante, el aprendizaje profundo cuenta con diferentes modelos que consisten en structuras especializadas según el tipo de información que se trate. Existen principalmente dos grandes grupos:
- **Redes Neuronales Convolucionales (CNNs)**: aplicadas principalmente a imágenes.
- **Redes Neuronales Recurrentes (RNNs)**: aplicadas a datos de tipo secuencial, como puedan ser el lenguaje o el sonido, en las cuales la información es recurrente (en una frase por ejemplo una palabra depende en mayor o menor medida de las palabras anteriores).

Espero que este post haya resultado útil para comprender un poco mejor en qué se basa el famoso Deep Learning. Si tienes cualquier duda o idea no dudes en poner un comentario.

**Que la tau os acompañe! ;)**
