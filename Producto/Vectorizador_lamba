import sys
import os
import json
import boto3
import pg8000.native
import shutil
import zipfile

# 1. FORZAR RUTA DE LA CAPA
sys.path.append('/opt/python')
from fastembed import TextEmbedding

# 2. CONFIGURACIÓN
BUCKET_NAME = "sara-repository-duoc" 
MODEL_ZIP_KEY = "modelo_sara.zip" 
TMP_BASE = "/tmp/fastembed_cache"

embedding_model = None

def ensure_model_is_ready():
    global embedding_model
    if embedding_model is not None:
        return

    # 1. Ruta plana que la librería pide según el log
    flat_path = os.path.join(TMP_BASE, "bge-small-en-v1.5")

    if not os.path.exists(flat_path):
        print(f"LOG: Instalando cerebro desde S3...")
        os.makedirs(TMP_BASE, exist_ok=True)
        s3 = boto3.client('s3')
        local_zip = "/tmp/temp_model.zip"
        
        s3.download_file(BUCKET_NAME, MODEL_ZIP_KEY, local_zip)
        extract_path = "/tmp/extrayendo"
        with zipfile.ZipFile(local_zip, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        os.remove(local_zip)

        # 2. BUSCADOR Y APLANADOR
        # Buscamos la carpeta que contiene el archivo .onnx real
        onnx_folder = None
        for root, dirs, files in os.walk(extract_path):
            if any(f.endswith(".onnx") for f in files):
                onnx_folder = root
                break
        
        if onnx_folder:
            print(f"LOG: Modelo encontrado en {onnx_folder}. Aplanando a {flat_path}...")
            os.makedirs(flat_path, exist_ok=True)
            for item in os.listdir(onnx_folder):
                shutil.move(os.path.join(onnx_folder, item), os.path.join(flat_path, item))
            shutil.rmtree(extract_path)
        else:
            raise Exception("No se encontró ningún archivo .onnx en el ZIP de S3.")

    # 3. INICIALIZACIÓN (Usando la ruta plana)
    print("LOG: Inicializando motor de IA local...")
    embedding_model = TextEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        cache_dir=TMP_BASE, # Buscará dentro de TMP_BASE/bge-small-en-v1.5
        local_files_only=True
    )

def get_embedding_local(text):
    try:
        ensure_model_is_ready()
        embeddings_generator = embedding_model.embed([text])
        vector = list(next(embeddings_generator))
        return [float(v) for v in vector]
    except Exception as e:
        print(f"Error en vectorización local: {e}")
        return None

def lambda_handler(event, context):
    print("LOG: Iniciando SARA Procesador")
    
    bucket = event['Records'][0]['s3']['bucket']['name']
    key = event['Records'][0]['s3']['object']['key']
    
    if key.endswith('.zip'):
        return {"status": "skipped"}

    s3_client = boto3.client('s3')
    response = s3_client.get_object(Bucket=bucket, Key=key)
    file_content = response['Body'].read().decode('utf-8')
    
    chunks = [file_content[i:i+1000] for i in range(0, len(file_content), 800)]
    print(f"Procesando {len(chunks)} fragmentos de {key}")

    try:
        conn = pg8000.native.Connection(
            user="postgres",
            host=os.environ['bd'],
            database="sara_db",
            password=os.environ['db_password']
        )

        for chunk in chunks:
            vector = get_embedding_local(chunk)
            if vector:
                print(f"DEBUG: Insertando vector (384 dimensiones)")
                
                # 1. SQL con parámetros nombrados
                sql = "INSERT INTO fragmentos_vectores (contenido, embedding, metadata) VALUES (:cont, :emb, :meta)"
                
                # 2. CONVERSIÓN CRÍTICA: Convertimos la lista a String con str(vector)
                # Esto garantiza que llegue a RDS con el formato "[v1, v2, ...]"
                conn.run(sql, 
                         cont=chunk, 
                         emb=str(vector), # <--- ¡ESTE ES EL CAMBIO CLAVE!
                         meta=json.dumps({"source": key}))
        
        conn.close()
        print("LOG: ¡SARA COMPLETADA! Todos los vectores guardados.")
        return {"status": "success", "file": key}
    except Exception as e:
        print(f"ERROR: {e}")
        raise e