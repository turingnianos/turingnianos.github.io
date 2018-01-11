---
layout: post
tags: redes-neuronales deep-learning inteligencia-artificial
date: 2018-01-11 9:30
thumbnail: /images/neural_net.jpeg
title: "Aprendizaje profundo: una introducción"
comments: true
published: false
---

Este post tiene como finalidad descubrir el intrigante mundo de las redes neuronales: una técnica de IA que en la última década ha generado resultados increíbles. Aprenderás por fin qué es eso del Deep Learning, como funciona una red neuronal y de qué modo se puede aplicar a diferentes problemas y situaciones del mundo real: detectar objetos en imágenes, generar música, traducir automáticamente de un idioma o vencer al ser humano en juegos como el Go o el Dota 2. Tranquilo, no hace falta ningún concepto previo de matemáticas o informática para entender este post.

<!--more-->

### Intersecciones: matemáticas, informática y biología
Si hay algo que me apasiona del conocimiento humano es cuando conceptos aparentemente inconexos se encuentran de repente. Si uno lo piensa bien, los mayores descubrimientos del ser humano resultan de pensar las cosas de un modo diferente. Para razonar usamos conceptos, de modo que esto lo podemos ver de otro modo: las más brillantes ideas surgen de asociar conceptos de forma original, diferente.

La [Inteligencia Artificial](https://turingnianos.github.io/2018/01/04/ia-que-es.html) se construye fundamentalmente a partir de conexiones de este tipo. Como el objetivo es lograr realizar de forma automática tareas propias del hombre y es el hombre el que trata de hacerlo, inevitablemente (para bien o para mal) aplica lo que percibe del mundo que le rodea. Es así como surgen por ejemplo los algoritmos genéticos (una técnica de aprendizaje automático): aplicando a la informática la idea de tener una población en la cual los que mejor se adaptan al entorno sobreviven y se reproducen dando lugar a variaciones en la especie. Como veis, esta técnica (que ya explicaremos en otro post) se basa en lo que en biología son dos de los mecanismos básicos de la evolución (selección natural y mutación).

Podríamos poner ejemplos de estas intersecciones con episodios como la manzana de Newton o el propio Eureka de Arquímedes, pero vamos a optar por otra más reciente: las redes neuronales artificiales.

### Redes neuronales:

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

{% include image.html 
   img="images/neurona.png" 
   title="Neurona" 
   caption="<b>Imagen 1:</b> <i>partes de una neurona.</i>" 
%}

Como podéis observar en la imagen, una neurona se compone de varias partes. Es importante entender las neuronas como vías por las cuales se transmite información (en forma de impulsos eléctricos). A nosotros nos interesa en particular lo siguiente:
* Las neuronas tienen un **cuerpo celular** llamado soma que contiene el núcleo.
* Las ramificaciones del cuerpo celular se llaman **dentritas** y se "conectan" con terminales nerviosas de otras neuronas.
* Hay también una extensión separada que suele ser más larga que recibe el nombre de **axón** y de la cual surgen las **terminales nerviosas** de la neurona.
Por tanto, para recapitular podemos decir que las neuronas tienen un cuerpo celular que tiene por un lado dentritas con las cuales otras neuronas se conectan con ella y terminales por las cuales ella se conecta a otras neuronas. Si ayuda, puedes ver las dentritas como diferentes entradas de USB que conectan con el cuerpo de la neurona y las terminales como una ramificación de diferentes USBs provenientes de un cable (axón) que sale desde el cuerpo.