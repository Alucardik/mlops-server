Так как файл модели занимает больше 100МБ места, то для его хранения используется Git Large File Storage (LFS). Для корректной работы проекта необходимо установить git lfs расширение перед клонированием репозитория - https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage.

Запуск:
```bash
pip install -r requirements.txt
python3 app.py
```