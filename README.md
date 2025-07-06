# face_processor
Pré processador de imagens para treinamento de LoRas

# Como usar

Instale os prerrequisitos:

```pip pip install opencv-python numpy```

Rode o programa:

```python face_processor.py```

Você deve especificar uma pasta de entrada que contenham fotos de boa qualidade, contendo o rosto de **uma** pessoa em cada foto. O programa irá centralizar o rosto da pessoa, horizontalmente, no centro da foto e, verticalmente, no terço superior da foto e, depois, irá cortar a imagem para que ela fique no formato 512x512 pixels, preservando ao máximo o que está no entorno do rosto posicionado de acordo com as instruções anteriores.

Especifique também a pasta de saída.

Esse programa foi testado no Ubuntu 24.04 com python3, usando venv. Em resumo, antes de instalar os prerrequisitos, você deve ter o ambiente virtual instalado dessa maneira:

```python -m venv venv # na pasta onde está o seu script em python```

```source ./venv/bin/activate```

# Como esse programa foi criado

Com vibe coding no claude.ai, usando três prompts:

**Prompt 1:**

I want you to build a python command line program that will do the following:

0. Ask the user to provide the input folder and the output folder  
    0.1. Check if the input folder contains images and provide the user with the image count  
        0.1.1. If the input folder contains images, just provide the image count and proceed; if not, exit with an error  
    0.2. If the output folder does not exists, create it.  
1. read all images from a given folder
2. analyze the faces on each one of the images and  
    2.1. position the face so it will be horizontally centered  
    2.2. position the face so the eyes will be positioned in the top most third part of the image  
    2.3. convert the image so it will be a square of 512x512 pixels, containing most of the original image around the re-positioned face  
    2.4. output the final image to a new folder.  

Note: this is a command to be used in the command line (bash), no web or no other kind of graphic interface.

**Prompt 2:**

Please take special attention to what I requested in item 2.3: containing (((most of the original image))) around the re-positioned face. It seems you mostly got the face, not the surroundings.

**Prompt 3:**

The results are even worse now. Almost anything of the original image is retained. Remember this:

2. analyze the faces on each one of the images and  
    2.1. position the face so it will be horizontally centered  
    2.2. position the face so the eyes will be positioned in the top most third part of the image  
    2.3. convert the image so it will be a square of 512x512 pixels, (((containing most of the original image around the re-positioned face)))  

**It worked after the third prompt!**
