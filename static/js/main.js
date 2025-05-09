document.addEventListener('DOMContentLoaded', function() {
    // Theme toggle functionality
    const themeToggle = document.querySelector('.theme-toggle');
    const body = document.body;
    
    // Check for saved theme preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        body.classList.add('dark-theme');
        themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
    }
    
    themeToggle.addEventListener('click', function() {
        body.classList.toggle('dark-theme');
        
        if (body.classList.contains('dark-theme')) {
            localStorage.setItem('theme', 'dark');
            themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
        } else {
            localStorage.setItem('theme', 'light');
            themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
        }
    });
    
    // Initialize IntersectionObserver for animations
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };
    
    const observer = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('animated');
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);
    
    // Observe all elements with animation classes
    document.querySelectorAll('.fade-in-io, .fade-in-up-io, .slide-in-right-io').forEach(el => {
        observer.observe(el);
    });
    
    // Staggered animations
    const staggerContainers = document.querySelectorAll('.stagger-container');
    staggerContainers.forEach(container => {
        const staggerItems = container.querySelectorAll('.stagger-item');
        
        const staggerObserver = new IntersectionObserver((entries, observer) => {
            if (entries[0].isIntersecting) {
                staggerItems.forEach((item, index) => {
                    setTimeout(() => {
                        item.classList.add('animated');
                    }, 100 * index);
                });
                staggerObserver.unobserve(container);
            }
        }, observerOptions);
        
        staggerObserver.observe(container);
    });
    
    // Counter animation
    const counterObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const counter = entry.target;
                const target = parseInt(counter.getAttribute('data-target'));
                const duration = 2000; // ms
                
                if (target > 100) {
                    // For large numbers, use a faster increment
                    const increment = Math.ceil(target / 100);
                    let current = 0;
                    
                    const timer = setInterval(() => {
                        current += increment;
                        counter.textContent = current;
                        counter.classList.add('animate');
                        
                        setTimeout(() => {
                            counter.classList.remove('animate');
                        }, 100);
                        
                        if (current >= target) {
                            counter.textContent = target;
                            clearInterval(timer);
                        }
                    }, duration / 100);
                } else {
                    // For small numbers, count one by one
                    let current = 0;
                    
                    const timer = setInterval(() => {
                        current += 1;
                        counter.textContent = current;
                        counter.classList.add('animate');
                        
                        setTimeout(() => {
                            counter.classList.remove('animate');
                        }, 100);
                        
                        if (current === target) {
                            clearInterval(timer);
                        }
                    }, duration / target);
                }
                
                counterObserver.unobserve(counter);
            }
        });
    }, { threshold: 0.5 });
    
    document.querySelectorAll('.counter').forEach(counter => {
        counterObserver.observe(counter);
    });
    
    // Fetch dataset info for the index page
    const statsContainer = document.querySelector('.stats-container');
    if (statsContainer) {
        fetch('/dataset/info')
            .then(response => response.json())
            .then(data => {
                // Update total images count if it exists
                const totalImagesCounter = document.querySelector('[data-target="total-images"]');
                if (totalImagesCounter && data.total_images) {
                    totalImagesCounter.setAttribute('data-target', data.total_images);
                }
            })
            .catch(error => console.error('Error:', error));
    }
    
    // Set current year in the footer
    const yearElement = document.getElementById('current-year');
    if (yearElement) {
        yearElement.textContent = new Date().getFullYear();
    }
});

// Utility function to format date
function formatDate(dateString) {
    const options = { year: 'numeric', month: 'long', day: 'numeric' };
    return new Date(dateString).toLocaleDateString(undefined, options);
}

// Utility function to format numbers
function formatNumber(num) {
    return num.toString().replace(/(\d)(?=(\d{3})+(?!\d))/g, '$1,');
}