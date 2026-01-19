(function() {
  'use strict';

  var state = {
    leaderboard: [],
    examples: [],
    search: '',
    sortDesc: true,
    venue: 'All',
    tag: 'All'
  };

  function toText(value) {
    if (value === undefined || value === null) {
      return '';
    }
    return String(value);
  }

  function toNumber(value) {
    var num = Number(value);
    return Number.isFinite(num) ? num : null;
  }

  function safeUrl(value) {
    if (!value || typeof value !== 'string') {
      return null;
    }
    var trimmed = value.trim();
    if (!trimmed) {
      return null;
    }
    try {
      var url = new URL(trimmed, document.baseURI);
      if (url.protocol === 'http:' || url.protocol === 'https:') {
        return url.href;
      }
      return null;
    } catch (error) {
      return null;
    }
  }

  function formatScore(value) {
    var num = toNumber(value);
    if (num === null) {
      return 'N/A';
    }
    if (Number.isInteger(num)) {
      return String(num);
    }
    return num.toFixed(2);
  }

  function setText(node, value) {
    node.textContent = value;
  }

  function clearChildren(node) {
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }

  function renderLeaderboard() {
    var tbody = document.getElementById('leaderboard-body');
    var emptyMessage = document.getElementById('leaderboard-empty');
    var countNode = document.getElementById('leaderboard-count');
    if (!tbody) {
      return;
    }

    var searchValue = state.search.toLowerCase();
    var filtered = state.leaderboard.filter(function(entry) {
      var method = toText(entry.method).toLowerCase();
      if (searchValue && method.indexOf(searchValue) === -1) {
        return false;
      }
      if (state.venue !== 'All') {
        var venue = toText(entry.venue).trim();
        if (venue !== state.venue) {
          return false;
        }
      }
      return true;
    });

    filtered.sort(function(a, b) {
      var scoreA = toNumber(a.score);
      var scoreB = toNumber(b.score);
      if (scoreA === null && scoreB === null) {
        return 0;
      }
      if (scoreA === null) {
        return 1;
      }
      if (scoreB === null) {
        return -1;
      }
      return state.sortDesc ? scoreB - scoreA : scoreA - scoreB;
    });

    clearChildren(tbody);

    filtered.forEach(function(entry, index) {
      var row = document.createElement('tr');

      var rankCell = document.createElement('td');
      setText(rankCell, String(index + 1));
      row.appendChild(rankCell);

      var methodCell = document.createElement('td');
      setText(methodCell, toText(entry.method) || 'Unknown');
      row.appendChild(methodCell);

      var scoreCell = document.createElement('td');
      setText(scoreCell, formatScore(entry.score));
      row.appendChild(scoreCell);

      var venueCell = document.createElement('td');
      setText(venueCell, toText(entry.venue).trim() || '—');
      row.appendChild(venueCell);

      var linksCell = document.createElement('td');
      var linksWrapper = document.createElement('div');
      linksWrapper.className = 'buttons are-small leaderboard-links';
      var hasLink = false;

      var paperUrl = safeUrl(entry.paper);
      if (paperUrl) {
        var paperLink = document.createElement('a');
        paperLink.className = 'button is-link is-light is-small';
        paperLink.href = paperUrl;
        paperLink.target = '_blank';
        paperLink.rel = 'noopener';
        setText(paperLink, 'Paper');
        linksWrapper.appendChild(paperLink);
        hasLink = true;
      }

      var codeUrl = safeUrl(entry.code);
      if (codeUrl) {
        var codeLink = document.createElement('a');
        codeLink.className = 'button is-link is-light is-small';
        codeLink.href = codeUrl;
        codeLink.target = '_blank';
        codeLink.rel = 'noopener';
        setText(codeLink, 'Code');
        linksWrapper.appendChild(codeLink);
        hasLink = true;
      }

      if (hasLink) {
        linksCell.appendChild(linksWrapper);
      } else {
        setText(linksCell, '—');
      }
      row.appendChild(linksCell);

      var notesCell = document.createElement('td');
      setText(notesCell, toText(entry.notes) || '—');
      row.appendChild(notesCell);

      tbody.appendChild(row);
    });

    if (countNode) {
      setText(countNode, String(filtered.length));
    }

    if (emptyMessage) {
      emptyMessage.classList.toggle('is-hidden', filtered.length !== 0);
    }
  }

  function updateVenueFilter() {
    var venueSelect = document.getElementById('leaderboard-venue');
    var venueWrap = document.getElementById('leaderboard-venue-wrap');
    if (!venueSelect || !venueWrap) {
      return;
    }

    var venues = {};
    state.leaderboard.forEach(function(entry) {
      var venue = toText(entry.venue).trim();
      if (venue) {
        venues[venue] = true;
      }
    });

    var venueList = Object.keys(venues).sort();
    clearChildren(venueSelect);

    if (venueList.length === 0) {
      venueWrap.classList.add('is-hidden');
      state.venue = 'All';
      return;
    }

    venueWrap.classList.remove('is-hidden');
    var allOption = document.createElement('option');
    allOption.value = 'All';
    setText(allOption, 'All');
    venueSelect.appendChild(allOption);

    venueList.forEach(function(venue) {
      var option = document.createElement('option');
      option.value = venue;
      setText(option, venue);
      venueSelect.appendChild(option);
    });

    venueSelect.value = state.venue;
  }

  function setupLeaderboardControls() {
    var searchInput = document.getElementById('leaderboard-search');
    var sortButton = document.getElementById('leaderboard-sort');
    var venueSelect = document.getElementById('leaderboard-venue');
    var lastUpdated = document.getElementById('leaderboard-last-updated');

    if (lastUpdated) {
      setText(lastUpdated, new Date().toLocaleString());
    }

    if (searchInput) {
      searchInput.addEventListener('input', function(event) {
        state.search = event.target.value || '';
        renderLeaderboard();
      });
    }

    if (sortButton) {
      sortButton.addEventListener('click', function() {
        state.sortDesc = !state.sortDesc;
        setText(sortButton, state.sortDesc ? 'Score: High to Low' : 'Score: Low to High');
        renderLeaderboard();
      });
    }

    if (venueSelect) {
      venueSelect.addEventListener('change', function(event) {
        state.venue = event.target.value || 'All';
        renderLeaderboard();
      });
    }
  }

  function renderExamples() {
    var grid = document.getElementById('examples-grid');
    var emptyMessage = document.getElementById('examples-empty');
    if (!grid) {
      return;
    }

    var filtered = state.examples.filter(function(example) {
      if (state.tag === 'All') {
        return true;
      }
      if (!Array.isArray(example.tags)) {
        return false;
      }
      return example.tags.indexOf(state.tag) !== -1;
    });

    clearChildren(grid);

    filtered.forEach(function(example) {
      var column = document.createElement('div');
      column.className = 'column is-half-tablet is-one-quarter-desktop';

      var card = document.createElement('div');
      card.className = 'card example-card';

      var cardImage = document.createElement('div');
      cardImage.className = 'card-image';
      var figure = document.createElement('figure');
      figure.className = 'image is-4by3';
      var img = document.createElement('img');
      var imageUrl = toText(example.image).trim();
      img.src = imageUrl ? new URL(imageUrl, document.baseURI).href : '';
      img.alt = toText(example.id) || 'Example';
      figure.appendChild(img);
      cardImage.appendChild(figure);
      card.appendChild(cardImage);

      var cardContent = document.createElement('div');
      cardContent.className = 'card-content';
      var content = document.createElement('div');
      content.className = 'content';

      var title = document.createElement('p');
      title.className = 'title is-6';
      setText(title, toText(example.id) || 'Example');
      content.appendChild(title);

      function addField(label, value) {
        var paragraph = document.createElement('p');
        var strong = document.createElement('strong');
        setText(strong, label + ': ');
        paragraph.appendChild(strong);
        var span = document.createElement('span');
        setText(span, toText(value) || '—');
        paragraph.appendChild(span);
        content.appendChild(paragraph);
      }

      addField('Input', example.input);
      addField('Output', example.output);
      if (example.gt) {
        addField('GT', example.gt);
      }

      if (Array.isArray(example.tags) && example.tags.length > 0) {
        var tagsWrapper = document.createElement('div');
        tagsWrapper.className = 'tags';
        example.tags.forEach(function(tag) {
          var tagSpan = document.createElement('span');
          tagSpan.className = 'tag is-rounded';
          setText(tagSpan, toText(tag));
          tagsWrapper.appendChild(tagSpan);
        });
        content.appendChild(tagsWrapper);
      }

      cardContent.appendChild(content);
      card.appendChild(cardContent);
      column.appendChild(card);
      grid.appendChild(column);
    });

    if (emptyMessage) {
      emptyMessage.classList.toggle('is-hidden', filtered.length !== 0);
    }
  }

  function updateTagFilters() {
    var tagControls = document.getElementById('examples-tag-controls');
    var tagContainer = document.getElementById('examples-tags');
    if (!tagControls || !tagContainer) {
      return;
    }

    var tags = {};
    state.examples.forEach(function(example) {
      if (!Array.isArray(example.tags)) {
        return;
      }
      example.tags.forEach(function(tag) {
        var cleanTag = toText(tag).trim();
        if (cleanTag) {
          tags[cleanTag] = true;
        }
      });
    });

    var tagList = Object.keys(tags).sort();
    clearChildren(tagContainer);

    if (tagList.length === 0) {
      tagControls.classList.add('is-hidden');
      state.tag = 'All';
      return;
    }

    tagControls.classList.remove('is-hidden');

    function createTagButton(tag) {
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'button is-small is-light tag-button';
      if (tag === state.tag) {
        button.classList.add('is-active');
      }
      setText(button, tag);
      button.addEventListener('click', function() {
        state.tag = tag;
        updateTagFilters();
        renderExamples();
      });
      return button;
    }

    tagContainer.appendChild(createTagButton('All'));
    tagList.forEach(function(tag) {
      tagContainer.appendChild(createTagButton(tag));
    });
  }

  function loadLeaderboard() {
    var url = new URL('data/leaderboard.json', document.baseURI);
    fetch(url)
      .then(function(response) {
        if (!response.ok) {
          throw new Error('Failed to load leaderboard');
        }
        return response.json();
      })
      .then(function(data) {
        state.leaderboard = Array.isArray(data) ? data : [];
        updateVenueFilter();
        renderLeaderboard();
      })
      .catch(function() {
        var emptyMessage = document.getElementById('leaderboard-empty');
        if (emptyMessage) {
          setText(emptyMessage, 'Unable to load leaderboard data.');
          emptyMessage.classList.remove('is-hidden');
        }
      });
  }

  function loadExamples() {
    var url = new URL('data/examples.json', document.baseURI);
    fetch(url)
      .then(function(response) {
        if (!response.ok) {
          throw new Error('Failed to load examples');
        }
        return response.json();
      })
      .then(function(data) {
        state.examples = Array.isArray(data) ? data : [];
        updateTagFilters();
        renderExamples();
      })
      .catch(function() {
        var emptyMessage = document.getElementById('examples-empty');
        if (emptyMessage) {
          setText(emptyMessage, 'Unable to load examples.');
          emptyMessage.classList.remove('is-hidden');
        }
      });
  }

  document.addEventListener('DOMContentLoaded', function() {
    setupLeaderboardControls();
    loadLeaderboard();
    loadExamples();
  });
})();
