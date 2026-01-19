document.addEventListener('DOMContentLoaded', function() {
  var burgers = Array.prototype.slice.call(document.querySelectorAll('.navbar-burger'), 0);

  burgers.forEach(function(burger) {
    burger.addEventListener('click', function() {
      var targetId = burger.dataset.target;
      var target = targetId ? document.getElementById(targetId) : document.querySelector('.navbar-menu');

      burger.classList.toggle('is-active');
      if (target) {
        target.classList.toggle('is-active');
      }
    });
  });

  var copyButton = document.getElementById('copy-bibtex');

  if (copyButton) {
    var originalLabel = copyButton.textContent;

    function setCopyState(success) {
      copyButton.textContent = success ? 'Copied!' : 'Copy failed';
      window.setTimeout(function() {
        copyButton.textContent = originalLabel;
      }, 2000);
    }

    function fallbackCopy(text) {
      var textarea = document.createElement('textarea');
      textarea.value = text;
      textarea.setAttribute('readonly', '');
      textarea.style.position = 'absolute';
      textarea.style.left = '-9999px';
      document.body.appendChild(textarea);
      textarea.select();
      try {
        var success = document.execCommand('copy');
        setCopyState(success);
      } catch (error) {
        setCopyState(false);
      }
      document.body.removeChild(textarea);
    }

    copyButton.addEventListener('click', function() {
      var targetId = copyButton.dataset.copyTarget;
      var target = targetId ? document.getElementById(targetId) : null;
      if (!target) {
        setCopyState(false);
        return;
      }

      var text = target.textContent || '';
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(function() {
          setCopyState(true);
        }).catch(function() {
          fallbackCopy(text);
        });
      } else {
        fallbackCopy(text);
      }
    });
  }

  function initCarousel(carousel) {
    var viewport = carousel.querySelector('.benchmark-carousel-viewport');
    var slides = Array.prototype.slice.call(carousel.querySelectorAll('[data-carousel-slide]'), 0);
    var caption = carousel.querySelector('[data-carousel-caption]');
    var prevButton = carousel.querySelector('.benchmark-carousel-prev');
    var nextButton = carousel.querySelector('.benchmark-carousel-next');

    if (!viewport || !prevButton || !nextButton || slides.length === 0) {
      return;
    }

    var index = 0;
    for (var i = 0; i < slides.length; i++) {
      if (!slides[i].hidden) {
        index = i;
        break;
      }
    }

    function render() {
      for (var j = 0; j < slides.length; j++) {
        slides[j].hidden = j !== index;
      }

      if (caption) {
        var title = slides[index].dataset && slides[index].dataset.title ? slides[index].dataset.title : 'Image';
        caption.textContent = title + ' (' + (index + 1) + '/' + slides.length + ')';
      }
    }

    function step(delta) {
      index = (index + delta + slides.length) % slides.length;
      render();
    }

    prevButton.addEventListener('click', function() {
      step(-1);
    });

    nextButton.addEventListener('click', function() {
      step(1);
    });

    viewport.addEventListener('keydown', function(event) {
      if (event.key === 'ArrowLeft') {
        event.preventDefault();
        step(-1);
      } else if (event.key === 'ArrowRight') {
        event.preventDefault();
        step(1);
      }
    });

    render();
  }

  var carousels = Array.prototype.slice.call(document.querySelectorAll('.benchmark-carousel'), 0);
  carousels.forEach(function(carousel) {
    initCarousel(carousel);
  });
});
