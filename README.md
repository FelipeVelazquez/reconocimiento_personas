# reconocimiento_personas

El siguiente programa es para poder identificar caracteristicas en imagenes, lo cual nos permite saber que es lo que el robot esta identificando.

El siguiente paquete contiene los archivos include, msg, scripts, src, CMakeList.txt,package.xml.
Los archivos mas importantes de este paquete estan dentro de la carpeta scripts el cual contiene los archivos en lenguaje de python.
Los requisitos necesarios para usarlo es tener instalado en Ubuntu el sitema operativo para robots (ROS) y opencv (se instala junto con ROS).

El progama 'reconocimiento_personas.py' opera de la siguiente manera: 
El promaga inicializa la camara de nuestra computadora para poder captar frame por frame, y despues de eso se lleva a un filtro de recorte el cual nos ayuda a poder reducir el area en el cual el programa analizara.
Con ayuda del programa 'talker.py' es el que nos proporcionara las coordenas necesarias para poder realizar el corte en la imagen.
Una vez hecho eso el programa continuara con el analisis respectivo y abriendo la pantalla para visualizar el analisis.

Es necesari tener el paquete dentro de tu espacio de trabajo.
Es necesario tener las imagenes para la base de datos la cual se almacenan dentro de la carpeta de tu espacio de trabajo.
Para poder hacer que funcione el programa primero es necesario hacer ejecutables los archivos 'talker.py' y 'reconocimiento_personas.py', para ello se entra a la termina y nos ubicamos dentro del la carpeta donde estan los archivos, en este caso la carpeta 'scipts' y con el siguinete comando haremos ejecutable $chmod 775 nombre_arcivo.py

Una vez de hacerlos ejecutables se procede a correr los programas para ver su funcionamiento.

Corremos el roscore y entramos a nuestro espacio de trabajo desde terminal ej. /home/catkin_ws$ 
Una vez ahi usamos el siguiente comando $rosrun nombre_paquete talker.py , es necesario que se inicie este programa primero ya que es el que nos mandara las coordenadas para el recorte, si no se corre este programa primero es probable que salga un error.
Despues de inicializar el talker.py se corre el programa de reconocimiento $rosrun nombre_paquete reconocimiento_personas.py

Con esto el programa de reconocimiento deberia esta corriendo perfectamente.
