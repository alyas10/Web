document.addEventListener('DOMContentLoaded', function() {
    const runAnalysisBtn = document.getElementById('run-analysis-btn');

    // Клик по всей карточке выбирает модель
    document.querySelectorAll('.model-card').forEach(card => {
        card.addEventListener('click', function(e) {
            // Если кликнули по инпуту, не переключаем радио (обработано через stopPropagation в HTML)
            const radio = this.querySelector('input[name="selected_model"]');
            if (radio) {
                radio.checked = true;
            }
        });
    });

    if (runAnalysisBtn) {
        runAnalysisBtn.addEventListener('click', function() {
            const selectedModelRadio = document.querySelector('input[name="selected_model"]:checked');

            if (!selectedModelRadio) {
                alert("Пожалуйста, выберите модель для анализа.");
                return;
            }

            const modelId = selectedModelRadio.value;
            const modelCard = document.querySelector(`[data-model-id="${modelId}"]`);

            // Собираем данные в объект
            const requestData = {
                algo: modelId,
                env: 'test', // или другое значение из вашего контекста
                params: {}
            };

            // Собираем гиперпараметры только из активной карточки
            const inputs = modelCard.querySelectorAll('.input-field');
            inputs.forEach(input => {
                if (input.name) {
                    requestData.params[input.name] = input.value;
                }
            });

            // Визуальная индикация загрузки
            runAnalysisBtn.disabled = true;
            const originalText = runAnalysisBtn.innerHTML;
            runAnalysisBtn.innerHTML = 'Обработка...';

            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestData)
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert("Ошибка сервера: " + data.error);
                } else {
                    console.log('Результаты:', data);
                    // Здесь можно вызвать модальное окно или обновить таблицу результатов
                    alert(`Анализ завершен успешно!\nМодель: ${data.model_used}\nКлассифицировано строк: ${data.count} \nПредсказанный класс: ${data.predicted_class}`);
                }
            })
            .catch(error => {
                console.error('Ошибка:', error);
                alert("Не удалось соединиться с сервером.");
            })
            .finally(() => {
                runAnalysisBtn.disabled = false;
                runAnalysisBtn.innerHTML = originalText;
            });
        });
    }
});