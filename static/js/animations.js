document.addEventListener('DOMContentLoaded', function() {
    // GSAP animations (if GSAP library is loaded)
    if (typeof gsap !== 'undefined') {
        // Hero section animations
        gsap.from('.hero-content h1', {
            duration: 1,
            y: 50,
            opacity: 0,
            ease: 'power3.out'
        });
        
        gsap.from('.hero-content p', {
            duration: 1,
            y: 30,
            opacity: 0,
            ease: 'power3.out',
            delay: 0.3
        });
        
        gsap.from('.hero-content .upload-btn', {
            duration: 1,
            y: 20,
            opacity: 0,
            ease: 'power3.out',
            delay: 0.6
        });
        
        gsap.from('.hero-image img', {
            duration: 1.2,
            x: 50,
            opacity: 0,
            ease: 'power3.out',
            delay: 0.3
        });
        
        // Staggered animations for feature cards
        gsap.from('.feature-card', {
            duration: 0.8,
            y: 50,
            opacity: 0,
            stagger: 0.2,
            ease: 'power3.out',
            scrollTrigger: {
                trigger: '.features',
                start: 'top 80%'
            }
        });
        
        // Plant cards animation
        gsap.from('.plant-card', {
            duration: 0.8,
            y: 30,
            opacity: 0,
            stagger: 0.2,
            ease: 'power3.out',
            scrollTrigger: {
                trigger: '.supported-plants',
                start: 'top 80%'
            }
        });
        
        // About page animations
        if (document.querySelector('.about-hero')) {
            gsap.from('.about-content h1', {
                duration: 1,
                y: 50,
                opacity: 0,
                ease: 'power3.out'
            });
            
            gsap.from('.about-content p', {
                duration: 1,
                y: 30,
                opacity: 0,
                ease: 'power3.out',
                delay: 0.3
            });
            
            gsap.from('.about-section', {
                duration: 0.8,
                y: 30,
                opacity: 0,
                stagger: 0.3,
                ease: 'power3.out',
                scrollTrigger: {
                    trigger: '.about-details',
                    start: 'top 80%'
                }
            });
        }
    }
    
    // Fallback animations for browsers without GSAP
    const animateFallback = function() {
        // Add animated class to fade-in elements
        document.querySelectorAll('.fade-in:not(.animated)').forEach(el => {
            el.classList.add('animated');
        });
        
        // Add animated class to visible fade-in-up elements
        document.querySelectorAll('.fade-in-up:not(.animated)').forEach(el => {
            if (isElementInViewport(el)) {
                el.classList.add('animated');
            }
        });
        
        // Add animated class to visible slide-in-right elements
        document.querySelectorAll('.slide-in-right:not(.animated)').forEach(el => {
            if (isElementInViewport(el)) {
                el.classList.add('animated');
            }
        });
    };
    
    // Check if element is in viewport
    function isElementInViewport(el) {
        const rect = el.getBoundingClientRect();
        return (
            rect.top <= (window.innerHeight || document.documentElement.clientHeight) &&
            rect.bottom >= 0
        );
    }
    
    // Run fallback animations on load and scroll
    animateFallback();
    window.addEventListener('scroll', animateFallback);
    
    // Add smooth scrolling to all anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                e.preventDefault();
                targetElement.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Particle animation for hero section (if canvas is available)
    const particleCanvas = document.getElementById('particle-canvas');
    if (particleCanvas) {
        const ctx = particleCanvas.getContext('2d');
        particleCanvas.width = window.innerWidth;
        particleCanvas.height = window.innerHeight;
        
        let particlesArray = [];
        
        // Particle class
        class Particle {
            constructor() {
                this.x = Math.random() * particleCanvas.width;
                this.y = Math.random() * particleCanvas.height;
                this.size = Math.random() * 3 + 1;
                this.speedX = Math.random() * 2 - 1;
                this.speedY = Math.random() * 2 - 1;
                this.color = '#4caf50';
            }
            
            update() {
                this.x += this.speedX;
                this.y += this.speedY;
                
                if (this.x > particleCanvas.width || this.x < 0) {
                    this.speedX = -this.speedX;
                }
                
                if (this.y > particleCanvas.height || this.y < 0) {
                    this.speedY = -this.speedY;
                }
            }
            
            draw() {
                ctx.fillStyle = this.color;
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
                ctx.fill();
            }
        }
        
        // Create particles
        function init() {
            particlesArray = [];
            let numberOfParticles = (particleCanvas.width * particleCanvas.height) / 15000;
            for (let i = 0; i < numberOfParticles; i++) {
                particlesArray.push(new Particle());
            }
        }
        
        // Animation loop
        function animate() {
            ctx.clearRect(0, 0, particleCanvas.width, particleCanvas.height);
            
            for (let i = 0; i < particlesArray.length; i++) {
                particlesArray[i].update();
                particlesArray[i].draw();
            }
            
            // Draw lines between particles
            connectParticles();
            
            requestAnimationFrame(animate);
        }
        
        // Connect particles with lines
        function connectParticles() {
            for (let a = 0; a < particlesArray.length; a++) {
                for (let b = a; b < particlesArray.length; b++) {
                    const dx = particlesArray[a].x - particlesArray[b].x;
                    const dy = particlesArray[a].y - particlesArray[b].y;
                    const distance = Math.sqrt(dx * dx + dy * dy);
                    
                    if (distance < 100) {
                        ctx.strokeStyle = 'rgba(76, 175, 80, ' + (1 - distance / 100) + ')';
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(particlesArray[a].x, particlesArray[a].y);
                        ctx.lineTo(particlesArray[b].x, particlesArray[b].y);
                        ctx.stroke();
                    }
                }
            }
        }
        
        // Handle window resize
        window.addEventListener('resize', function() {
            particleCanvas.width = window.innerWidth;
            particleCanvas.height = window.innerHeight;
            init();
        });
        
        // Initialize and start animation
        init();
        animate();
    }
});