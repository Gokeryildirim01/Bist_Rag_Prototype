# 📈 BIST Yapay Zeka Finans ve Teknik Analiz Asistanı (ADEL Prototipi)

Bu proje, Borsa İstanbul (BIST) verilerini **RAG (Retrieval-Augmented Generation)** mimarisi, **LLM (Groq / Llama 3)**, **Matematiksel İntegral Hesaplamaları** ve **Gradio** arayüzü ile birleştiren bir yapay zeka finansal asistan prototipidir.

## 🚀 Özellikler
- **Teknik Analiz & İntegral Alanı:** Hisse fiyatı ve hareketli ortalamalar (SMA) arasındaki farkın integralini (`scipy.integrate`) hesaplayarak trend yoğunluğunu görselleştirir.
- **RAG Mimarisi:** Canlı borsa verilerini ve şirket bilgilerini FAISS vektör veritabanına gömerek LLM'e aktarır.
- **Canlı Web Arayüzü:** Gradio ile kullanıcı dostu grafik ve analitik yanıt ekranı sunar.

## 🛠️ Kullanılan Teknolojiler
- **Dil:** Python
- **LLM & Framework:** LangChain, Groq API (`llama3-8b-8192`)
- **Vektör Veritabanı:** FAISS, HuggingFace Embeddings
- **Veri & Matematik:** `yfinance`, `scipy`, `numpy`, `matplotlib`
- **Arayüz:** Gradio

## ⚙️ Kurulum ve Çalıştırma

1. Repoyu klonlayın:
   ```bash
   git clone [https://github.com/Gokeryildirim01/Bist_Rag_Prototype.git](https://github.com/Gokeryildirim01/Bist_Rag_Prototype.git)
   cd Bist_Rag_Prototype

   pip install gradio langchain langchain-groq langchain-community yfinance matplotlib scipy faiss-cpu
   export GROQ_API_KEY="your_groq_api_key"
   python main.py '''

   ⚠️ YASAL UYARI / DISCLAIMER
Bu sitede/projede yer alan yatırım bilgi, yorum ve tavsiyeleri yatırım danışmanlığı kapsamında değildir. Sunulan içerikler yalnızca teknik analiz, matematiksel modelleme ve yapay zeka eğitim prototipi amacıyla üretilmektedir. Hiçbir şekilde alım-satım yönlendirmesi veya yatırım tavsiyesi içermez.
