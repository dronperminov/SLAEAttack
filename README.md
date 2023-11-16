# SLAEAttack
Реализация атаки на полносвязную сеть с использованием методов линейной алгебры

## Начало работы
Установите все необходимые зависимости:

```python
pip install -r requirements.txt
```

## Запуск обучения
Для обучения используется скрипт `train.py`:

```python
python3 train.py
```

## Запуск веб интерфейса атаки
Для запуска используется скрипт `main.py`, запускающий через uvicorn FactAPI приложение, доступное по адресу `localhost:8931`:

```python
python3 main.py
```
