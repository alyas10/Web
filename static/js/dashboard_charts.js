/**
 * Dashboard Charts Module
 * Интерактивные дашборды для страницы анализа
 * Использует Chart.js для визуализации данных
 */

class DashboardCharts {
  constructor() {
    this.charts = {};
    this.data = null;
    this.init();
  }

  /**
   * Инициализация всех графиков
   */
  init() {
    // Ждем загрузки DOM
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => this.loadAndInit());
    } else {
      this.loadAndInit();
    }
  }

  /**
   * Загрузка данных и инициализация
   */
  async loadAndInit() {
    try {
      // Пытаемся загрузить данные из session или API
      this.data = await this.fetchDashboardData();

      if (this.data) {
        this.initAllCharts();
        this.updateSummaryCards();
        this.checkDemoMode();
      } else {
        this.showNoDataMessage();
      }
    } catch (error) {
      console.error('Error loading dashboard data:', error);
      this.showErrorMessage(error.message);
    }
  }

  /**
   * Получение данных дашборда
   */
  async fetchDashboardData() {
    // Проверяем, есть ли данные в глобальной переменной (переданы из Flask)
    if (window.dashboardData) {
      return window.dashboardData;
    }

    // Пытаемся получить через API
    try {
      const response = await fetch('/api/dashboard-data');
      if (response.ok) {
        return await response.json();
      }
    } catch (e) {
      console.log('API data not available, using demo mode');
    }

    // Возвращаем демо-данные если нет реальных
    return this.getDemoData();
  }

  /**
   * Демо-данные для тестирования
   */
  getDemoData() {
    return {
      total_events: 15847,
      threats_detected: 342,
      safe_traffic: 15505,
      model_used: 'LightGBM',
      is_demo: true,
      class_distribution: {
        'Benign': 12450,
        'DoS/DDoS': 1820,
        'Intrusion': 987,
        'Anomaly': 456,
        'Port Scan': 134
      },
      model_metrics: [
        { name: 'LightGBM', accuracy: 0.964, f1: 0.952 },
        { name: 'XGBoost', accuracy: 0.951, f1: 0.943 },
        { name: 'Random Forest', accuracy: 0.938, f1: 0.921 },
        { name: 'Isolation Forest', accuracy: 0.892, f1: 0.876 }
      ],
      analysis_history: [
        { date: '2026-02-07', threats: 87 },
        { date: '2026-02-06', threats: 124 },
        { date: '2026-02-05', threats: 56 },
        { date: '2026-02-04', threats: 93 },
        { date: '2026-02-03', threats: 71 }
      ],
      top_threats: [
        { type: 'DoS/DDoS', count: 1820 },
        { type: 'Intrusion', count: 987 },
        { type: 'Anomaly', count: 456 },
        { type: 'Port Scan', count: 134 }
      ]
    };
  }

  /**
   * Обновление карточек сводки
   */
  updateSummaryCards() {
    if (!this.data) return;

    this.animateValue('ds-total', 0, this.data.total_events);
    this.animateValue('ds-threats', 0, this.data.threats_detected);
    this.animateValue('ds-safe', 0, this.data.safe_traffic);

    const modelEl = document.getElementById('ds-model');
    if (modelEl) {
      modelEl.textContent = this.data.model_used || '—';
    }
  }

  /**
   * Анимация числовых значений
   */
  animateValue(elementId, start, end, duration = 1000) {
    const element = document.getElementById(elementId);
    if (!element) return;

    const range = end - start;
    const startTime = performance.now();

    const update = (currentTime) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);

      // Easing function (easeOutQuart)
      const ease = 1 - Math.pow(1 - progress, 4);
      const current = Math.floor(start + (range * ease));

      element.textContent = current.toLocaleString('ru-RU');

      if (progress < 1) {
        requestAnimationFrame(update);
      }
    };

    requestAnimationFrame(update);
  }

  /**
   * Инициализация всех графиков
   */
  initAllCharts() {
    this.initTrafficDonut();
    this.initModelComparison();
    this.initThreatHistory();
    this.initTopThreats();
  }

  /**
   * Donut chart: Распределение классов трафика
   */
  initTrafficDonut() {
    const ctx = document.getElementById('chartDonut');
    if (!ctx || !this.data?.class_distribution) return;

    const labels = Object.keys(this.data.class_distribution);
    const values = Object.values(this.data.class_distribution);
    const total = values.reduce((a, b) => a + b, 0);

    // Цвета для классов
    const colors = {
      'Benign': '#059669',
      'DoS/DDoS': '#dc2626',
      'Intrusion': '#f97316',
      'Anomaly': '#eab308',
      'Port Scan': '#8b5cf6',
      'default': '#6b7280'
    };

    const backgroundColors = labels.map(label =>
      colors[label] || colors['default']
    );

    this.charts.donut = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: labels,
        datasets: [{
          data: values,
          backgroundColor: backgroundColors,
          borderWidth: 0,
          hoverOffset: 10
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '65%',
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(31, 41, 55, 0.95)',
            titleColor: '#dfe1e5',
            bodyColor: '#9ea2aa',
            borderColor: '#374151',
            borderWidth: 1,
            padding: 12,
            displayColors: true,
            callbacks: {
              label: (context) => {
                const value = context.parsed;
                const percentage = ((value / total) * 100).toFixed(1);
                return `${value.toLocaleString()} (${percentage}%)`;
              }
            }
          }
        }
      }
    });

    // Создаем кастомный легенд
    this.createDonutLegend(labels, values, total, colors);
  }

  /**
   * Создание кастомной легенды для donut chart
   */
  createDonutLegend(labels, values, total, colors) {
    const legendContainer = document.getElementById('donutLegend');
    if (!legendContainer) return;

    legendContainer.innerHTML = labels.map((label, index) => {
      const value = values[index];
      const percentage = ((value / total) * 100).toFixed(1);
      const color = colors[label] || colors['default'];

      return `
        <div style="display:flex;align-items:center;justify-content:space-between;padding:0.5rem 0;border-bottom:1px solid rgba(55,65,81,0.5);">
          <div style="display:flex;align-items:center;gap:0.5rem;">
            <div style="width:12px;height:12px;border-radius:2px;background:${color};"></div>
            <span style="font-size:0.8125rem;color:var(--es-text-primary);">${label}</span>
          </div>
          <div style="text-align:right;">
            <div style="font-size:0.8125rem;font-weight:600;color:var(--es-text-primary);">${value.toLocaleString()}</div>
            <div style="font-size:0.6875rem;color:var(--es-text-secondary);">${percentage}%</div>
          </div>
        </div>
      `;
    }).join('');
  }

  /**
   * Bar chart: Сравнение моделей по Accuracy и F1-score
   */
  initModelComparison() {
    const ctx = document.getElementById('chartModels');
    if (!ctx || !this.data?.model_metrics) return;

    const labels = this.data.model_metrics.map(m => m.name);
    const accuracyData = this.data.model_metrics.map(m => m.accuracy * 100);
    const f1Data = this.data.model_metrics.map(m => m.f1 * 100);

    this.charts.models = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: 'Accuracy %',
            data: accuracyData,
            backgroundColor: 'rgba(0, 119, 204, 0.8)',
            borderColor: 'rgba(0, 119, 204, 1)',
            borderWidth: 1,
            borderRadius: 4,
            barPercentage: 0.6
          },
          {
            label: 'F1-Score %',
            data: f1Data,
            backgroundColor: 'rgba(168, 85, 247, 0.8)',
            borderColor: 'rgba(168, 85, 247, 1)',
            borderWidth: 1,
            borderRadius: 4,
            barPercentage: 0.6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: {
              color: '#9ea2aa',
              font: { size: 11 },
              usePointStyle: true
            }
          },
          tooltip: {
            backgroundColor: 'rgba(31, 41, 55, 0.95)',
            titleColor: '#dfe1e5',
            bodyColor: '#9ea2aa',
            borderColor: '#374151',
            borderWidth: 1,
            padding: 12,
            callbacks: {
              label: (context) => `${context.dataset.label}: ${context.parsed.toFixed(2)}%`
            }
          }
        },
        scales: {
          y: {
            beginAtZero: false,
            min: 80,
            max: 100,
            grid: {
              color: 'rgba(55, 65, 81, 0.5)'
            },
            ticks: {
              color: '#9ea2aa',
              callback: (value) => value + '%'
            }
          },
          x: {
            grid: {
              display: false
            },
            ticks: {
              color: '#9ea2aa'
            }
          }
        }
      }
    });
  }

  /**
   * Line chart: История анализов
   */
  initThreatHistory() {
    const ctx = document.getElementById('chartHistory');
    if (!ctx || !this.data?.analysis_history) return;

    const labels = this.data.analysis_history.map(h => h.date);
    const threatData = this.data.analysis_history.map(h => h.threats);

    this.charts.history = new Chart(ctx, {
      type: 'line',
      data: {
        labels: labels,
        datasets: [{
          label: 'Угроз обнаружено',
          data: threatData,
          borderColor: '#dc2626',
          backgroundColor: 'rgba(220, 38, 38, 0.1)',
          borderWidth: 2,
          fill: true,
          tension: 0.4,
          pointBackgroundColor: '#dc2626',
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
          pointRadius: 4,
          pointHoverRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(31, 41, 55, 0.95)',
            titleColor: '#dfe1e5',
            bodyColor: '#9ea2aa',
            borderColor: '#374151',
            borderWidth: 1,
            padding: 12,
            callbacks: {
              label: (context) => `Угроз: ${context.parsed}`
            }
          }
        },
        scales: {
          y: {
            grid: {
              color: 'rgba(55, 65, 81, 0.5)'
            },
            ticks: {
              color: '#9ea2aa'
            }
          },
          x: {
            grid: {
              display: false
            },
            ticks: {
              color: '#9ea2aa'
            }
          }
        }
      }
    });
  }

  /**
   * Horizontal bar chart: Топ угроз
   */
  initTopThreats() {
    const ctx = document.getElementById('chartTopThreats');
    if (!ctx || !this.data?.top_threats) return;

    const labels = this.data.top_threats.map(t => t.type);
    const counts = this.data.top_threats.map(t => t.count);

    // Градиент цветов от красного к оранжевому
    const colors = counts.map((_, i) => {
      const ratio = i / counts.length;
      return `rgba(220, 38, 38, ${1 - ratio * 0.5})`;
    });

    this.charts.topThreats = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Количество',
          data: counts,
          backgroundColor: colors,
          borderColor: '#dc2626',
          borderWidth: 1,
          borderRadius: 4,
          barPercentage: 0.7
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: 'rgba(31, 41, 55, 0.95)',
            titleColor: '#dfe1e5',
            bodyColor: '#9ea2aa',
            borderColor: '#374151',
            borderWidth: 1,
            padding: 12,
            callbacks: {
              label: (context) => `Обнаружено: ${context.parsed}`
            }
          }
        },
        scales: {
          x: {
            grid: {
              color: 'rgba(55, 65, 81, 0.5)'
            },
            ticks: {
              color: '#9ea2aa'
            }
          },
          y: {
            grid: {
              display: false
            },
            ticks: {
              color: '#9ea2aa',
              font: { size: 11 }
            }
          }
        }
      }
    });
  }

  /**
   * Проверка режима демо-данных
   */
  checkDemoMode() {
    const badge = document.getElementById('demoBadge');
    if (badge && this.data?.is_demo) {
      badge.style.display = 'block';
    }
  }

  /**
   * Показать сообщение об отсутствии данных
   */
  showNoDataMessage() {
    const container = document.getElementById('interactiveDashboards');
    if (!container) return;

    container.innerHTML = `
      <div class="card" style="padding:2rem;text-align:center;">
        <svg style="width:48px;height:48px;color:var(--es-text-secondary);margin-bottom:1rem;"
             fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
        </svg>
        <p style="color:var(--es-text-primary);font-weight:600;margin-bottom:0.5rem;">
          Нет данных для отображения
        </p>
        <p style="color:var(--es-text-secondary);font-size:0.875rem;">
          Запустите анализ проекта чтобы увидеть интерактивные дашборды
        </p>
      </div>
    `;
  }

  /**
   * Показать сообщение об ошибке
   */
  showErrorMessage(message) {
    const container = document.getElementById('interactiveDashboards');
    if (!container) return;

    container.innerHTML = `
      <div class="card" style="padding:2rem;text-align:center;border-color:var(--es-danger);">
        <svg style="width:48px;height:48px;color:var(--es-danger);margin-bottom:1rem;"
             fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
        </svg>
        <p style="color:var(--es-danger);font-weight:600;margin-bottom:0.5rem;">
          Ошибка загрузки данных
        </p>
        <p style="color:var(--es-text-secondary);font-size:0.875rem;">
          ${message}
        </p>
      </div>
    `;
  }

  /**
   * Обновление данных графиков
   */
  updateData(newData) {
    this.data = newData;
    this.initAllCharts();
    this.updateSummaryCards();
  }

  /**
   * Уничтожение графиков (для очистки памяти)
   */
  destroy() {
    Object.values(this.charts).forEach(chart => {
      if (chart && typeof chart.destroy === 'function') {
        chart.destroy();
      }
    });
    this.charts = {};
  }
}

// Экспорт для глобального использования
window.DashboardCharts = DashboardCharts;

// Авто-инициализация при загрузке страницы
if (document.getElementById('interactiveDashboards')) {
  new DashboardCharts();
}