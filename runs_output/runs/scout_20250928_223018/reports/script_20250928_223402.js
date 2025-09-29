// Comprehensive Business Report JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initializeAnimations();
    initializeCharts();
    initializeScrollEffects();
    initializeInteractiveElements();
});

function initializeAnimations() {
    // Intersection Observer for fade-in animations
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe all sections
    document.querySelectorAll('.section').forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(30px)';
        section.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(section);
    });
}

function initializeCharts() {
    // Initialize revenue chart if canvas exists
    const revenueCanvas = document.getElementById('revenueChart');
    if (revenueCanvas) {
        createRevenueChart(revenueCanvas);
    }
}

function createRevenueChart(canvas) {
    const ctx = canvas.getContext('2d');
    
    // Chart data
    const data = {
        labels: ['Year 1', 'Year 2', 'Year 3', 'Year 4', 'Year 5'],
        datasets: [{
            label: 'Revenue Projections',
            data: [500000, 2000000, 5000000, 10000000, 20000000],
            backgroundColor: 'rgba(37, 99, 235, 0.1)',
            borderColor: 'rgba(37, 99, 235, 1)',
            borderWidth: 2,
            fill: true
        }]
    };

    // Chart configuration
    const config = {
        type: 'line',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return '$' + (value / 1000000).toFixed(1) + 'M';
                        }
                    }
                }
            },
            animation: {
                duration: 2000,
                easing: 'easeInOutQuart'
            }
        }
    };

    // Create chart
    new Chart(ctx, config);
}

function initializeScrollEffects() {
    // Parallax effect for hero section
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        const heroSection = document.querySelector('.hero-section');
        
        if (heroSection) {
            heroSection.style.transform = `translateY(${scrolled * 0.5}px)`;
        }
    });

    // Progress bar for reading progress
    createReadingProgressBar();
}

function createReadingProgressBar() {
    // Create progress bar element
    const progressBar = document.createElement('div');
    progressBar.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 0%;
        height: 3px;
        background: linear-gradient(90deg, #2563eb, #4f46e5);
        z-index: 1000;
        transition: width 0.3s ease;
    `;
    document.body.appendChild(progressBar);

    // Update progress on scroll
    window.addEventListener('scroll', () => {
        const winScroll = document.body.scrollTop || document.documentElement.scrollTop;
        const height = document.documentElement.scrollHeight - document.documentElement.clientHeight;
        const scrolled = (winScroll / height) * 100;
        progressBar.style.width = scrolled + '%';
    });
}

function initializeInteractiveElements() {
    // Add hover effects to cards
    document.querySelectorAll('.summary-card, .gap-card, .recommendation-card').forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
            this.style.boxShadow = '0 10px 25px rgba(0, 0, 0, 0.15)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
            this.style.boxShadow = '0 4px 6px rgba(0, 0, 0, 0.1)';
        });
    });

    // Add click effects to timeline items
    document.querySelectorAll('.timeline-item').forEach(item => {
        item.addEventListener('click', function() {
            this.style.transform = 'scale(1.02)';
            setTimeout(() => {
                this.style.transform = 'scale(1)';
            }, 200);
        });
    });

    // Add smooth scrolling for internal links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// Utility functions
function formatCurrency(amount) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(amount);
}

function animateCounter(element, start, end, duration) {
    let startTimestamp = null;
    const step = (timestamp) => {
        if (!startTimestamp) startTimestamp = timestamp;
        const progress = Math.min((timestamp - startTimestamp) / duration, 1);
        const current = Math.floor(progress * (end - start) + start);
        element.textContent = current.toLocaleString();
        if (progress < 1) {
            window.requestAnimationFrame(step);
        }
    };
    window.requestAnimationFrame(step);
}

// Initialize counter animations for stats
document.addEventListener('DOMContentLoaded', function() {
    const statNumbers = document.querySelectorAll('.stat-number');
    statNumbers.forEach(stat => {
        const finalValue = parseInt(stat.textContent);
        if (!isNaN(finalValue)) {
            animateCounter(stat, 0, finalValue, 2000);
        }
    });
});

// Export functions for potential external use
window.BusinessReport = {
    initializeAnimations,
    initializeCharts,
    initializeScrollEffects,
    initializeInteractiveElements,
    formatCurrency,
    animateCounter
};
