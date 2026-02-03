# 📘 Technical Report  
## Design and Implementation of a Persian AI-NLP Service with OCR Capability  
### طراحی و پیاده‌سازی سرویس هوشمند پردازش زبان طبیعی فارسی با قابلیت OCR

---

## 1. Introduction | مقدمه

### فارسی
با گسترش حجم اسناد دیجیتال و نیاز روزافزون به استخراج، تحلیل و خلاصه‌سازی اطلاعات متنی، طراحی سامانه‌های هوشمند پردازش زبان طبیعی (NLP) به‌ویژه برای زبان فارسی اهمیت ویژه‌ای یافته است.  
هدف این پروژه، پیاده‌سازی یک **سرویس جامع و ماژولار NLP مبتنی بر هوش مصنوعی** است که بتواند اسناد متنی و غیرمتنی (تصاویر و PDFهای اسکن‌شده) را دریافت کرده، متن آن‌ها را استخراج، پاک‌سازی، پردازش و خلاصه‌سازی نماید.

### English
With the rapid growth of digital documents and the increasing demand for automatic text understanding, analysis, and summarization, intelligent Natural Language Processing (NLP) systems—especially for low-resource languages such as Persian—have become increasingly important.  
The objective of this project is to design and implement a **modular AI-based NLP service** capable of ingesting both editable and non-editable documents, extracting textual content, preprocessing it, and producing structured and summarized outputs through an API.

---

## 2. Project Overview | نمای کلی پروژه

### فارسی
سرویس **NLP-AI** به‌صورت ماژولار طراحی شده و شامل قابلیت‌های زیر است:
- اصلاح و پاک‌سازی متن (Text Correction)
- استخراج متن از اسناد (OCR)
- خلاصه‌سازی متون (Summarization)
- قابلیت توسعه برای طبقه‌بندی و استخراج اطلاعات

طراحی اولیه پروژه، مبنای توسعه نسخه عملیاتی و پیشرفته فعلی قرار گرفته است.

### English
The **NLP-AI Service** is designed as a modular system providing the following capabilities:
- Text correction and normalization
- Optical Character Recognition (OCR)
- Text summarization
- Extensible architecture for classification and information extraction

---

## 3. System Architecture | معماری سیستم

### فارسی
معماری پروژه به‌صورت لایه‌ای و توسعه‌پذیر پیاده‌سازی شده است:

app/
├── api/ # API Routes (FastAPI)
├── core/ # Configuration & Settings
├── services/ # OCR & NLP Services
├── models/ # Data & ML Models
└── db/ # Database Layer


ویژگی‌های کلیدی:
- FastAPI
- تنظیمات مبتنی بر `.env`
- استفاده از `pydantic_settings`
- پایگاه داده پیش‌فرض SQLite

### English
The system follows a layered and extensible architecture with clear separation of concerns and environment-based configuration management.

---

## 4. OCR Module | ماژول OCR

### فارسی
ماژول OCR از ورودی‌های زیر پشتیبانی می‌کند:
- تصاویر (PNG, JPG, …)
- PDF (اسکن‌شده یا متنی)
- فایل‌های متنی (TXT)

ویژگی‌ها:
- تعیین بازه صفحات
- پیش‌نمایش هوشمند PDFهای حجیم
- OCR چندزبانه (فارسی + انگلیسی)
- پاک‌سازی متن با LLM
- اصلاح فاصله‌ها و نیم‌فاصله (ZWNJ)

### English
The OCR module supports multiple formats and includes multilingual OCR, performance-aware PDF processing, and LLM-based text cleanup.

---

## 5. Text Preprocessing & Cleaning | پیش‌پردازش متن

### فارسی
- حذف نویز OCR
- اصلاح غلط‌های املایی
- تنظیم فاصله‌ها و نیم‌فاصله فارسی
- حفظ معنای متن

### English
The preprocessing pipeline improves text quality through noise removal, spelling correction, and orthographic normalization.

---

## 6. Text Chunking & Summarization | بخش‌بندی و خلاصه‌سازی

### فارسی
- تقسیم متن به بخش‌های منطقی
- جلوگیری از قطع جملات
- تولید خلاصه برای هر بخش

روش‌ها:
1. Rule-based
2. آماری (TextRank)
3. مبتنی بر مدل‌های زبانی (HuggingFace / LLM)

### English
Long documents are chunked safely and summarized using multiple strategies, including statistical and transformer-based models.

---

## 7. Model Selection Rationale | انتخاب مدل

### فارسی
مدل **mT5 Multilingual XLSum** به‌دلیل:
- عمومی بودن
- مصرف منابع کمتر
- بهینه‌سازی برای خلاصه‌سازی چندزبانه
- عملکرد مناسب در فارسی

انتخاب شد.

### English
mT5 XLSum was selected due to open access, efficiency, and strong multilingual summarization performance.

---

## 8. API Design | طراحی API

### فارسی

POST /ocr


ورودی:
- فایل
- بازه صفحات (اختیاری)
- فعال‌سازی پاک‌سازی متن

خروجی:
- نوع فایل
- متن استخراج‌شده
- وضعیت پاک‌سازی

### English
The OCR endpoint returns structured, API-ready results for downstream NLP tasks.

---

## 9. Future Work | کارهای آتی

### فارسی
- ذخیره نتایج در دیتابیس
- پردازش async
- تست‌های واحد
- Dockerization
- بهبود OCR با مدل‌های پیش‌آموزش‌دیده

### English
Future work includes persistence, async processing, testing, containerization, and advanced OCR models.

---
