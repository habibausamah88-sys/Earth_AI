# طريقة تجربة البروجكت

1. افتحي Google Colab واختاري Runtime > Change runtime type > T4 GPU.
2. افتحي START_COLAB.ipynb من File > Upload notebook. ارفعي EarthLens_AI_Fixed.zip لما الخلية تطلبه.
3. شغلي خلايا Quick test بالترتيب. الكود بينزل الداتا ويعمل Audit ويدرب عينة صغيرة. الـ Smoke Test للتأكد من التشغيل فقط، مش نتيجة نهائية.
4. لو الاختبار نجح، شغلي Full training وبعدين التجارب حسب ترتيب الخلايا. متشغليش Final test إلا بعد انتهاء التجارب واختيار الموديل بالـ validation.
5. النتائج في experiments/، الموديلات في models/، والـ demo موجود في app/app.py. حملي نسخة النتائج قبل انتهاء جلسة Colab.

ملحوظة: لا يوجد تدريب فعلي في النسخة المضغوطة. التقرير والسلايدز لازم يتعملوا من النتائج الحقيقية. راجعي docs/CHANGES.md و report/README.md.
