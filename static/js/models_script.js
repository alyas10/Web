document.addEventListener('DOMContentLoaded', function() {
    // === 1. Инициализация элементов ===
    const runAnalysisBtn = document.getElementById('run-analysis-btn');
    const modal = document.getElementById('vizModal');

    // Элементы внутри модального окна
    const vizElements = {
        loading: document.getElementById('vizLoading'),
        content: document.getElementById('vizContent'),
        error: document.getElementById('vizError'),
        title: document.getElementById('modalTitle'),
        //imgPipeline: document.getElementById('imgPipeline'),
        imgImportance: document.getElementById('imgImportance'),
        imgTree: document.getElementById('imgTree'),
        metaInfo: document.getElementById('metaInfo')
    };

    // === 2. Обработка карточек моделей ===

    // Клик по всей карточке для выбора модели
    document.querySelectorAll('.model-card').forEach(card => {
        card.addEventListener('click', function(e) {
            // Если клик был по кнопке визуализации или инпуту параметра - не переключаем радио
            if (e.target.closest('.viz-btn') || e.target.closest('.input-field')) {
                return;
            }

            const radio = this.querySelector('input[name="selected_model"]');
            if (radio) {
                radio.checked = true;
                // Генерируем событие change, если нужно обновлять стили программно
                radio.dispatchEvent(new Event('change', { bubbles: true }));
            }
        });
    });

    // === 3. Запуск анализа (Predict) ===

    if (runAnalysisBtn) {
        runAnalysisBtn.addEventListener('click', async function() {
            const selectedRadio = document.querySelector('input[name="selected_model"]:checked');

            if (!selectedRadio) {
                showToast("Пожалуйста, выберите модель для анализа", "warning");
                return;
            }

            const modelId = selectedRadio.value;
            const modelCard = document.querySelector(`[data-model-id="${modelId}"]`);

            // Сбор данных
            const requestData = {
                algo: modelId,
                env: 'test',
            };

            // Индикация загрузки
            toggleLoadingState(runAnalysisBtn, true);

            try {
                const response = await fetch('/models/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestData)
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.error || `Ошибка сервера (${response.status})`);
                }

                // Успешный результат - показываем красивое уведомление
                showToast(`✅ Анализ завершен! Обработано строк: ${data.count}. Переход на Dashboard...`, "success");

                // Перенаправление на страницу Dashboard через небольшую паузу
                setTimeout(() => {
                    window.location.href = '/dashboard';
                }, 1500); // Ждем 1.5 секунды чтобы пользователь увидел уведомление

            } catch (error) {
                console.error('Analysis Error:', error);
                showToast("❌ Ошибка при запуске анализа: " + error.message, "error");
            } finally {
                toggleLoadingState(runAnalysisBtn, false);
            }
        });
    }

    // === 4. Логика Модального Окна (Визуализация) ===

    /**
     * Открывает модальное окно и загружает данные графиков
     */
    window.openVizModal = function(modelId, modelName) {
        if (!modal) return;

        // Показываем модалку и блокируем скролл страницы
        modal.style.display = 'flex';
        document.body.style.overflow = 'hidden';
        modal.setAttribute('aria-hidden', 'false');

        // Сброс состояния UI
        if (vizElements.title) vizElements.title.textContent = '📊 Визуализация: ' + modelName;
        updateVizState('loading');

        // Запрос к API визуализации
        fetch(`/models/api/viz/${modelId}`)
            .then(res => {
                if (!res.ok) throw new Error(`Ошибка сервера (${res.status})`);
                return res.json();
            })
            .then(data => {
                if (!data.success) throw new Error(data.error || 'Данные не найдены');

                //ленивая загрузка
                window.vizPayload = window.vizPayload || {};
                window.vizPayload.pipeline = data.pipeline || null;
                window.vizPayload.pipelineType = data.pipeline_type || null;
                window.vizPayload.importance = data.importance || null;
                window.vizPayload.tree = data.tree || null;

                // 1. Загружаем график ТОЛЬКО активного таба (ленивая загрузка)
              const activeTab = document.querySelector('.viz-tab.active')?.dataset.tab;

            /*   if (vizElements.imgPipeline && data.pipeline) {
              //vizElements.imgPipeline.src = 'image/png;base64,' + data.pipeline;
              vizElements.imgPipeline.src = 'data:image/png;base64,' + data.pipeline;
               vizElements.imgPipeline.classList.add('loaded');
               }*/
            /*   if (activeTab === 'pipeline' && data.pipeline && data.pipeline_type === 'html') {
        const container = document.getElementById('pipelineContainer');
        if (container) {
            container.innerHTML = data.pipeline;
         }
          }*/
             if (activeTab === 'importance' && vizElements.imgImportance && data.importance) {
              vizElements.imgImportance.src = 'data:image/png;base64,' + data.importance;
              vizElements.imgImportance.classList.add('loaded');
               }
             if (activeTab === 'tree' && vizElements.imgTree && data.tree) {
             vizElements.imgTree.src = 'data:image/png;base64,' + data.tree;
             vizElements.imgTree.classList.add('loaded');
               }
                // 2. Мета-информация (гиперпараметры)
                if (vizElements.metaInfo && data.info) {
                    vizElements.metaInfo.innerHTML = Object.entries(data.info)
                        .map(([key, value]) => `
                            <div class="viz-meta-item">
                                <span class="viz-meta-label">${formatLabel(key)}</span>
                                <span class="viz-meta-value">${value}</span>
                            </div>
                        `).join('');
                }

                updateVizState('content');
            })
            .catch(err => {
                console.error('Viz Load Error:', err);
                updateVizState('error', err.message);
            });
    };

    /**
     * Закрывает модальное окно
     */
    window.closeModal = function() {
        if (!modal) return;

        modal.style.display = 'none';
        document.body.style.overflow = '';
        modal.setAttribute('aria-hidden', 'true');

        // Очистка ресурсов для предотвращения утечек памяти
        if (vizElements.imgImportance) vizElements.imgImportance.src = '';
        if (vizElements.imgTree) vizElements.imgTree.src = '';
        //if (vizElements.imgPipeline) vizElements.imgPipeline.src = '';
    };

    window.downloadImage = function(imgId, filename) {
    const img = document.getElementById(imgId);
    if (!img || !img.src) return;

    const link = document.createElement('a');
    link.href = img.src;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    };

    // === 5. Вспомогательные функции ===

    function toggleLoadingState(btn, isLoading) {
        if (isLoading) {
            btn.dataset.originalHtml = btn.innerHTML;
            btn.disabled = true;
            btn.innerHTML = `<span class="spinner"></span> Обработка...`;
        } else {
            btn.innerHTML = btn.dataset.originalHtml;
            btn.disabled = false;
        }
    }

    function updateVizState(state, errorMessage = '') {
        if (vizElements.loading) vizElements.loading.style.display = state === 'loading' ? 'flex' : 'none';
        if (vizElements.content) vizElements.content.style.display = state === 'content' ? 'block' : 'none';
        if (vizElements.error) {
            vizElements.error.style.display = state === 'error' ? 'block' : 'none';
            if (errorMessage) vizElements.error.textContent = '⚠️ ' + errorMessage;
        }
    }

    function formatLabel(key) {
        const labels = {
            'trees': 'Кол-во деревьев',
            'n_estimators': 'Итераций',
            'max_depth': 'Глубина дерева',
            'learning_rate': 'Скорость обучения',
            'num_leaves': 'Листьев',
            'features': 'Признаков',
            'accuracy': 'Точность'
        };
        return labels[key] || key;
    }

    // === 6. Обработчики событий (Слушатели) ===

    // Переключение вкладок в модальном окне
     // Находим элементы управления
const tabs = document.querySelectorAll('.viz-tab');
const tabPanes = document.querySelectorAll('.tab-pane');
const titleElement = document.getElementById('currentTabTitle');
const downloadBtn = document.getElementById('downloadBtn');
const footerHint = document.getElementById('vizCardFooter');

const tabTitles = {
  pipeline: 'Структура Pipeline',
  importance: 'Важность признаков (Feature Importance)',
  tree: 'Структура дерева решений',
  info: 'Информация'
};

// сюда будем класть base64, но src в img ставить только при открытии таба
let vizPayload = { importance: null, tree: null };

function activateTab(targetId) {
  // 1) активная кнопка
  tabs.forEach(t => t.classList.toggle('active', t.dataset.tab === targetId));

  // 2) активная панель
  tabPanes.forEach(pane => {
    pane.classList.toggle('active', pane.id === `tab-${targetId}`);
  });

  // 3) заголовок
  if (titleElement) titleElement.textContent = tabTitles[targetId] || '';

  // 4) скачать (скрываем на info)
  if (downloadBtn) downloadBtn.style.display = (targetId === 'info') ? 'none' : 'block';

  // 5) футер только на дереве
  if (footerHint) footerHint.style.display = (targetId === 'tree') ? 'block' : 'none';

  // 6) LAZY: ставим src только когда вкладка реально открыта
  /*if (targetId === 'pipeline' && vizElements.imgPipeline && !vizElements.imgPipeline.src && window.vizPayload?.pipeline) {
    //vizElements.imgPipeline.src = 'image/png;base64,' + window.vizPayload.pipeline;
    vizElements.imgPipeline.src = 'data:image/png;base64,' + window.vizPayload.pipeline;
}*/
if (targetId === 'pipeline' && window.vizPayload?.pipeline && window.vizPayload.pipelineType === 'html') {
        const container = document.getElementById('pipelineContainer');
        if (container) {
            container.innerHTML = window.vizPayload.pipeline;
        }
    }
  if (targetId === 'importance' && vizElements.imgImportance && !vizElements.imgImportance.src && window.vizPayload?.importance) {
    vizElements.imgImportance.src = 'data:image/png;base64,' + window.vizPayload.importance;
}
if (targetId === 'tree' && vizElements.imgTree  && window.vizPayload?.tree) {
//&& !vizElements.imgTree.src
    vizElements.imgTree.src = 'data:image/png;base64,' + window.vizPayload.tree;
}
  resetAllZoom();
}

// клики по табам
tabs.forEach(tab => {
  tab.addEventListener('click', function() {
    const targetId = this.dataset.tab;   // <-- ВАЖНО: используем targetId
    activateTab(targetId);
  });
});

    // Закрытие по клику на фон
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) window.closeModal();
        });
    }

    // Закрытие по Escape
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && modal?.style.display === 'flex') {
            window.closeModal();
        }
    });

    // Доступность: обработка кнопок визуализации в карточках
    document.querySelectorAll('.viz-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            window.openVizModal(btn.dataset.modelId, btn.dataset.modelName);
        });
    });
const zoomConfig = {
    minScale: 1,
    maxScale: 5,
    step: 0.25
};

// Хранилище состояния зума для каждого графика
const zoomState = {
    importance: { scale: 1, x: 0, y: 0, isDragging: false, startX: 0, startY: 0 },
    tree: { scale: 1, x: 0, y: 0, isDragging: false, startX: 0, startY: 0 }
};


/**
 * Сбросить зум всех графиков
 */
function resetAllZoom() {
    ['importance', 'tree'].forEach(key => {
        const wrapper = document.getElementById(`wrapper${key.charAt(0).toUpperCase() + key.slice(1)}`);
        const img = document.getElementById(`img${key.charAt(0).toUpperCase() + key.slice(1)}`);
        if (img) {
            img.style.transform = 'scale(1) translate(0px, 0px)';
            if (wrapper) wrapper.classList.remove('zoomed');
        }
        zoomState[key] = { scale: 1, x: 0, y: 0, isDragging: false, startX: 0, startY: 0 };
    });
}


/**
 * Инициализировать зум для конкретного графика
 */
function initZoom(imgId, wrapperId, stateKey) {
    const wrapper = document.getElementById(wrapperId);
    const img = document.getElementById(imgId);
    const state = zoomState[stateKey];

    if (!wrapper || !img) return;

    // Сброс трансформации
    function resetTransform() {
        state.scale = 1;
        state.x = 0;
        state.y = 0;
        img.style.transform = 'scale(1) translate(0px, 0px)';
        wrapper.classList.remove('zoomed');
    }
    // Зум колесом мыши
    wrapper.addEventListener('wheel', (e) => {
        e.preventDefault();
        const delta = e.deltaY < 0 ? zoomConfig.step : -zoomConfig.step;
        state.scale = Math.min(zoomConfig.maxScale, Math.max(zoomConfig.minScale, state.scale + delta));

        if (state.scale > 1) {
            wrapper.classList.add('zoomed');
            wrapper.style.cursor = 'grab';
        } else {
            resetTransform();
        }
        applyTransform();
    }, { passive: false });

    // Применение трансформации
    function applyTransform() {
        img.style.transform = `scale(${state.scale}) translate(${state.x}px, ${state.y}px)`;
    }

     wrapper.addEventListener('mousedown', (e) => {
        if (state.scale <= 1) return;
        state.isDragging = true;
        state.startX = e.clientX - state.x;
        state.startY = e.clientY - state.y;
        wrapper.style.cursor = 'grabbing';
        e.preventDefault();
    });

     document.addEventListener('mousemove', (e) => {
        if (!state.isDragging || state.scale <= 1) return;
        state.x = e.clientX - state.startX;
        state.y = e.clientY - state.startY;
        applyTransform();
    });

    document.addEventListener('mouseup', () => {
        if (state.isDragging) {
            state.isDragging = false;
            if (state.scale > 1) wrapper.style.cursor = 'grab';
        }
    });

    // Двойной клик — сброс зума
    wrapper.addEventListener('dblclick', (e) => {
        e.preventDefault();
        resetTransform();
    });

        // Сохраняем resetTransform для внешнего вызова
    wrapper._resetZoom = resetTransform;
}

/**
 * Инициализировать зум для всех графиков в модалке
 */
function initModalZoom() {
    initZoom('imgImportance', 'wrapperImportance', 'importance');
    initZoom('imgTree', 'wrapperTree', 'tree');
}

// 1. После успешной загрузки данных в openVizModal — инициализируем зум
const originalOpenVizModal = window.openVizModal;
window.openVizModal = function(modelId, modelName) {
    originalOpenVizModal?.(modelId, modelName);

    // Инициализируем зум с небольшой задержкой (после отрисовки)
    setTimeout(() => {
        initModalZoom();
    }, 150);
};

// 2. При закрытии модалки — сбрасываем зум
const originalCloseModal = window.closeModal;
window.closeModal = function() {
    resetAllZoom();
    originalCloseModal?.();
};

// 3. Обработчик кнопки "Скачать" (учитывает активный таб)
window.handleDownload = function() {
    const activeTab = document.querySelector('.viz-tab.active')?.dataset.tab;
    const imgId = activeTab === 'importance' ? 'imgImportance' : 'imgTree';
    const filename = activeTab === 'importance' ? 'feature_importance.png' : 'decision_tree.png';
    downloadImage(imgId, filename);
};

// 4. Глобальная функция сброса (для отладки)
window.resetZoom = resetAllZoom;
});
