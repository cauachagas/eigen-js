/**
 * Client-side Hash Router for Eigen.js Demo & Docs
 */

import { renderHomeView } from './views/home.js';
import { renderDocsView } from './views/docs.js';
import { renderBenchmarkView } from './views/benchmark.js';

export const routes = {
  home: {
    hash: '#/',
    title: 'Eigen JS',
    documentTitle: 'Eigen.js - Linear Algebra for WebAssembly',
    render: container => renderHomeView(container),
  },
  matrix: {
    hash: '#/matrix',
    title: 'Matrix',
    documentTitle: 'Matrix - Eigen.js Documentation',
    render: container => renderDocsView(container, 'Matrix'),
  },
  solvers: {
    hash: '#/solvers',
    title: 'Solvers',
    documentTitle: 'Solvers - Eigen.js Documentation',
    render: container => renderDocsView(container, 'Solvers'),
  },
  decompositions: {
    hash: '#/decompositions',
    title: 'Decompositions',
    documentTitle: 'Decompositions - Eigen.js Documentation',
    render: container => renderDocsView(container, 'Decompositions'),
  },
  benchmark: {
    hash: '#/benchmark',
    title: 'Benchmark',
    documentTitle: 'Benchmarks of Linear Algebra JavaScript Libraries - Eigen.js',
    render: container => renderBenchmarkView(container),
  },
};

export function getRouteKeyFromHash(hash) {
  const cleanHash = (hash || window.location.hash || '').replace(/^#\/?/, '').toLowerCase();

  if (!cleanHash || cleanHash === 'home') return 'home';
  if (cleanHash === 'matrix') return 'matrix';
  if (cleanHash === 'solvers') return 'solvers';
  if (cleanHash === 'decompositions') return 'decompositions';
  if (cleanHash === 'benchmark' || cleanHash === 'benchmarks') return 'benchmark';

  return 'home';
}

export function initRouter(container, onNavigateCallback) {
  function handleRouteChange() {
    const routeKey = getRouteKeyFromHash(window.location.hash);
    const route = routes[routeKey] || routes.home;

    // Update browser title
    document.title = route.documentTitle;

    // Render view
    container.innerHTML = '';
    route.render(container);

    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'auto' });

    // Update active state in navigation
    document.querySelectorAll('.drawer-item').forEach(item => {
      const itemRoute = item.getAttribute('data-route');
      if (itemRoute === routeKey) {
        item.classList.add('active');
      } else {
        item.classList.remove('active');
      }
    });

    // Notify callback
    if (typeof onNavigateCallback === 'function') {
      onNavigateCallback(routeKey, route);
    }
  }

  // Listen to hashchange
  window.addEventListener('hashchange', handleRouteChange);

  // Initial render
  handleRouteChange();
}
