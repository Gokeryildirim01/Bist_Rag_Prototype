import os
import traceback
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import yfinance as yf
import gradio as gr

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq

# =====================================================================
# 1. ORTAM VE API KEY AYARI
# =====================================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")

# =====================================================================
# 2. CANLI BIST VERİSİ ÇEKME & DOKÜMAN OLUŞTURMA
# =====================================================================
try:
    adel = yf.Ticker("ADEL.IS")
    info = adel.info

    kar_marji = info.get('profitMargins', 0)
    faaliyet_marji = info.get('operatingMargins', 0)

    metin = f"""
    Şirket Adı: {info.get('longName', 'ADEL Kalemcilik')}
    Faaliyet Sektörü: {info.get('sector', 'Tüketim Ürünleri')}
    Şirket Özeti: {info.get('longBusinessSummary', 'Bilgi bulunamadı.')}

    Borsa Göstergeleri:
    - Anlık Hisse Fiyatı: {info.get('currentPrice', info.get('previousClose', 'Bilinmiyor'))} TL
    - En Yüksek (52 Hafta): {info.get('fiftyTwoWeekHigh', 'Bilinmiyor')} TL
    - En Düşük (52 Hafta): {info.get('fiftyTwoWeekLow', 'Bilinmiyor')} TL
    - Piyasa Değeri: {info.get('marketCap', 'Bilinmiyor')} TL
    - Fiyat-Kazanç Oranı (F/K): {info.get('trailingPE', 'Bilinmiyor')}

    Finansal Performans ve Marjlar:
    - Net Kâr Marjı: %{kar_marji * 100:.2f}
    - Faaliyet Kâr Marjı: %{faaliyet_marji * 100:.2f}
    - Toplam Gelir: {info.get('totalRevenue', 'Bilinmiyor')} TL
    - Net Dönem Kârı: {info.get('netIncomeToCommon', 'Bilinmiyor')} TL
    """

    docs = [Document(page_content=metin)]
    print(" Canlı borsa verileri başarıyla yüklendi!")
except Exception as e:
    print(" Veri çekme hatası:", e)
    docs = [Document(page_content="ADEL şirket verileri şu an çekilemedi.")]

# =====================================================================
# 3. METİN BÖLME VE FAISS VEKTÖR DEPOSU
# =====================================================================
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(splits, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# =====================================================================
# 4. LLM VE RAG ZİNCİRİ
# =====================================================================
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama3-8b-8192",  
    temperature=0.4,
    max_tokens=1000
)

template = """Sen uzman, dost canlısı ve analitik bir Finans Asistanısın. 
Görevin, sana sağlanan verileri, teknik göstergeleri ve integral hesaplamalarını kullanarak kullanıcının sorularını yanıtlamaktır.

Erişebildiğin Güncel Veriler:
{context}

Soru: {question}

Yanıt Kuralları:
1. Doğrudan verilere odaklan, "Ben yatırım danışmanı değilim" gibi gereksiz cümlelerle başlama.
2. Kısa ve uzun vadeli (Long/Short term) durum analizini açıkla.
3. İntegral alanının ve teknik verilerin ne anlama geldiğini yorumla.
4. Teknik seviyelere dayanarak senaryo bazlı işlem stratejisi önerisi sun.

Cevap:"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(documents):
    return "\n\n".join(doc.page_content for doc in documents)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print(" RAG Zinciri ve LLM Mimarisi Hazır")

# =====================================================================
# 5. İNTEGRAL HESAPLAMA VE GRAFİK ÇİZİMİ
# =====================================================================
def analiz_ve_grafik_olustur(ticker_symbol="ADEL.IS"):
    stock = yf.Ticker(ticker_symbol)
    df = stock.history(period="1y")

    if df.empty:
        return "Veri bulunamadı", None

    # Hareketli Ortalamalar (SMA)
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()

    guncel_fiyat = df['Close'].iloc[-1]
    sma20_son = df['SMA20'].iloc[-1]
    sma50_son = df['SMA50'].iloc[-1]

    # INTEGRAL HESABI: Son 30 günlük fiyat eğrisinin altındaki alan
    fiyatlar_son30 = df['Close'].tail(30).values
    gunler = np.arange(len(fiyatlar_son30))
    fiyat_integrali = integrate.simpson(fiyatlar_son30, x=gunler)
    ortalama_alan = fiyat_integrali / 30

    kisa_vade_durum = "YUKARI (Bullish)" if guncel_fiyat > sma20_son else "AŞAĞI (Bearish)"
    uzun_vade_durum = "YUKARI (Bullish)" if sma20_son > sma50_son else "AŞAĞI (Bearish)"

    metin_ekstra = f"""
    Teknik Göstergeler ve Integral Hesaplamaları:
    - Kısa Vadeli Ortalama (20 SMA): {sma20_son:.2f} TL
    - Orta/Uzun Vadeli Ortalama (50 SMA): {sma50_son:.2f} TL
    - Son 30 Günlük Fiyat Integral Alanı: {fiyat_integrali:.2f} (Günlük Ortalama Momentum: {ortalama_alan:.2f})
    - Kısa Vade Trend Yönü (Short Term): {kisa_vade_durum}
    - Uzun Vade Trend Yönü (Long Term): {uzun_vade_durum}
    """

    # Grafik Çizimi
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df.index[-90:], df['Close'][-90:], label="Fiyat", color="blue")
    ax.plot(df.index[-90:], df['SMA20'][-90:], label="20 SMA (Kısa Vade)", color="orange", linestyle="--")
    ax.plot(df.index[-90:], df['SMA50'][-90:], label="50 SMA (Uzun Vade)", color="green", linestyle="--")
    ax.fill_between(df.index[-30:], df['Close'][-30:], color='skyblue', alpha=0.4, label="Integral Alanı (30 Gün)")

    ax.set_title(f"{ticker_symbol} Trend ve Integral Grafiği")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return metin_ekstra, fig

# =====================================================================
# 6. GRADIO ARAYÜZÜ VE UYGULAMA BAŞLATICI
# =====================================================================
def sohbet_ve_grafik(soru):
    try:
        cevap = rag_chain.invoke(soru)
        _, guncel_grafik = analiz_ve_grafik_olustur("ADEL.IS")
        return cevap, guncel_grafik

    except Exception as e:
        hata_detayi = traceback.format_exc()
        print("--- HATA DETAYI ---")
        print(hata_detayi)
        
        hata_mesaji = f"⚠️ İşlem sırasında bir hata oluştu:\n{str(e)}"
        return hata_mesaji, None

with gr.Blocks() as demo:
    gr.Markdown("# 📈 Yapay Zeka Borsa ve Grafik Asistanı 'Sadece Adel için prototip'")
    
    with gr.Row():
        with gr.Column(scale=1):
            soru_kutusu = gr.Textbox(
                label="Sorunuzu Yazın", 
                placeholder="Örn: ADEL için teknik analiz ve strateji yorumun nedir?"
            )
            btn = gr.Button("Analiz Et", variant="primary")
            cevap_kutusu = gr.Textbox(
                label="Yapay Zeka Analizi ve Stratejisi", 
                lines=12
            )
        
        with gr.Column(scale=1):
            grafik_kutusu = gr.Plot(label="Fiyat ve Integral Alan Grafiği")
            
    btn.click(
        fn=sohbet_ve_grafik, 
        inputs=soru_kutusu, 
        outputs=[cevap_kutusu, grafik_kutusu]
    )
    
    # Sabit Yasal Uyarı Metni
    gr.Markdown(
        """
        ---
        <small>⚠️ **YASAL UYARI:** Bu sitede yer alan yatırım bilgi, yorum ve tavsiyeleri **yatırım danışmanlığı kapsamında değildir.** 
        Sunulan içerikler yalnızca teknik analiz, matematiksel modelleme ve yapay zeka eğitim prototipi amacıyla üretilmektedir. 
        Hiçbir şekilde alım-satım yönlendirmesi veya yatırım tavsiyesi içermez.</small>
        """
    )

if __name__ == "__main__":
    demo.launch(share=True, debug=True)
