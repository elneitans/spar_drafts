"""
Módulo para manejar llamadas a modelos LLM (Llama 2, Qwen, etc.).

Este módulo centraliza toda la lógica de carga y generación de modelos LLM,
permitiendo que otros módulos los usen sin necesidad de conocer los detalles
de implementación de cada modelo.
"""
# Imports para modelos LLM (opcionales, solo se cargan si se usa)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# Cache global para los modelos (se cargan una sola vez)
_llama_model = None
_llama_tokenizer = None
_qwen_model = None
_qwen_tokenizer = None
_llama3_model = None
_llama3_tokenizer = None


def load_llama2_model(model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
    """
    Carga el modelo Llama 2 7B y su tokenizer.
    
    El modelo se carga una sola vez y se reutiliza para todas las llamadas.
    Requiere autenticación con Hugging Face (token) para acceder al modelo.
    
    Args:
        model_name: Nombre del modelo en Hugging Face (default: Llama-2-7b-chat-hf)
        
    Returns:
        Tupla (tokenizer, model)
        
    Raises:
        ImportError: Si transformers no está instalado
        RuntimeError: Si hay problemas cargando el modelo
    """
    global _llama_model, _llama_tokenizer
    
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers y torch no están instalados. "
            "Instala con: pip install transformers torch accelerate"
        )
    
    # Si ya está cargado, retornar
    if _llama_model is not None and _llama_tokenizer is not None:
        return _llama_tokenizer, _llama_model
    
    print(f"Cargando modelo {model_name}...")
    print("⚠️  Nota: Requiere token de Hugging Face. Configura con: huggingface-cli login")
    
    try:
        # Cargar tokenizer
        _llama_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Cargar modelo
        _llama_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Si no hay GPU, mover a CPU
        if not torch.cuda.is_available():
            _llama_model = _llama_model.to("cpu")
        
        print(f"✓ Modelo {model_name} cargado correctamente")
        if torch.cuda.is_available():
            print(f"  Usando GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  Usando CPU (puede ser lento)")
        
        return _llama_tokenizer, _llama_model
        
    except Exception as e:
        raise RuntimeError(
            f"Error cargando modelo {model_name}: {str(e)}\n"
            f"Asegúrate de tener un token de Hugging Face configurado."
        ) from e


def generate_with_llama2(prompt: str, max_new_tokens: int = 350) -> str:
    """
    Genera un resumen usando Llama 2 7B.
    
    Args:
        prompt: Prompt completo para el modelo
        max_new_tokens: Número máximo de tokens nuevos a generar (default: 500,
                       suficiente para ~300-400 palabras en español)
        
    Returns:
        Texto del resumen generado
    """
    tokenizer, model = load_llama2_model()
    
    # Preparar el prompt para Llama 2 Chat
    # Llama 2 Chat usa un formato especial con tokens de sistema/usuario
    system_message = "Eres un asistente experto en resumir CVs para reclutadores chilenos en español."
    formatted_prompt = f"<s>[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n{prompt} [/INST]"
    
    # Tokenizar (truncar si es muy largo)
    max_input_length = 8192
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length
    )
    
    # Mover inputs al dispositivo correcto
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Configurar pad_token si no existe
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Generar
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Decodificar solo los tokens generados (no el prompt completo)
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Limpiar el texto generado
    generated_text = generated_text.strip()
    
    # Si está vacío, intentar decodificar todo y extraer la parte nueva
    if not generated_text:
        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if "[/INST]" in full_text:
            generated_text = full_text.split("[/INST]")[-1].strip()
        else:
            generated_text = full_text[len(formatted_prompt):].strip()
    
    # Liberar memoria de CUDA si está disponible
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return generated_text


def load_meta_llama3_model(model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
    """
    Carga el modelo Meta Llama 3.1 8B (Instruct) y su tokenizer.

    Se comporta similar a `load_llama2_model`: carga una sola vez y reutiliza.
    """
    global _llama3_model, _llama3_tokenizer

    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers y torch no están instalados. "
            "Instala con: pip install transformers torch accelerate"
        )

    if _llama3_model is not None and _llama3_tokenizer is not None:
        return _llama3_tokenizer, _llama3_model

    print(f"Cargando modelo {model_name}...")
    print("⚠️  Nota: Requiere token de Hugging Face. Configura con: huggingface-cli login")

    try:
        _llama3_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        _llama3_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

        if not torch.cuda.is_available():
            _llama3_model = _llama3_model.to("cpu")

        print(f"✓ Modelo {model_name} cargado correctamente")
        if torch.cuda.is_available():
            print(f"  Usando GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  Usando CPU (puede ser lento)")

        return _llama3_tokenizer, _llama3_model

    except Exception as e:
        raise RuntimeError(
            f"Error cargando modelo {model_name}: {str(e)}\n"
            f"Asegúrate de tener un token de Hugging Face configurado."
        ) from e


def generate_with_meta_llama3(prompt: str, max_new_tokens: int = 350) -> str:
    """
    Genera texto usando Meta-Llama-3.1-8B-Instruct.
    """
    tokenizer, model = load_meta_llama3_model()

    system_message = "Eres un asistente experto en resumir CVs para reclutadores chilenos en español."
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    # ✅ Llama 3 Instruct: usar chat template nativo, NO [INST]/<<SYS>>
    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    max_input_length = 8192
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length
    )

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )

    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return generated_text


def load_qwen_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    """
    Carga el modelo Qwen 2.5 7B y su tokenizer.
    
    El modelo se carga una sola vez y se reutiliza para todas las llamadas.
    
    Args:
        model_name: Nombre del modelo en Hugging Face (default: Qwen2.5-7B-Instruct)
        
    Returns:
        Tupla (tokenizer, model)
        
    Raises:
        ImportError: Si transformers no está instalado
        RuntimeError: Si hay problemas cargando el modelo
    """
    global _qwen_model, _qwen_tokenizer
    
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError(
            "transformers y torch no están instalados. "
            "Instala con: pip install transformers torch accelerate"
        )
    
    # Si ya está cargado, retornar
    if _qwen_model is not None and _qwen_tokenizer is not None:
        return _qwen_tokenizer, _qwen_model
    
    print(f"Cargando modelo {model_name}...")
    
    try:
        # Cargar tokenizer
        _qwen_tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Cargar modelo
        _qwen_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        # Si no hay GPU, mover a CPU
        if not torch.cuda.is_available():
            _qwen_model = _qwen_model.to("cpu")
        
        print(f"✓ Modelo {model_name} cargado correctamente")
        if torch.cuda.is_available():
            print(f"  Usando GPU: {torch.cuda.get_device_name(0)}")
        else:
            print(f"  Usando CPU (puede ser lento)")
        
        return _qwen_tokenizer, _qwen_model
        
    except Exception as e:
        raise RuntimeError(
            f"Error cargando modelo {model_name}: {str(e)}"
        ) from e


def generate_with_qwen(prompt: str, max_new_tokens: int = 350) -> str:
    """
    Genera un resumen usando Qwen 2.5 7B.
    
    Args:
        prompt: Prompt completo para el modelo
        max_new_tokens: Número máximo de tokens nuevos a generar (default: 500,
                       suficiente para ~300-400 palabras en español)
        
    Returns:
        Texto del resumen generado
    """
    tokenizer, model = load_qwen_model()
    
    # Preparar el prompt para Qwen 2.5 Instruct
    # Qwen usa un formato con mensajes de sistema y usuario
    messages = [
        {
            "role": "system",
            "content": "Eres un asistente experto en resumir CVs para reclutadores chilenos en español."
        },
        {
            "role": "user",
            "content": prompt
        }
    ]
    
    # Aplicar template de chat
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Tokenizar
    max_input_length = 8192
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_length
    )
    
    # Mover inputs al dispositivo correcto
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Configurar pad_token si no existe
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Generar
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
    
    # Decodificar solo los tokens generados
    input_length = inputs['input_ids'].shape[1]
    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Limpiar el texto generado
    generated_text = generated_text.strip()
    
    # Liberar memoria de CUDA si está disponible
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return generated_text

