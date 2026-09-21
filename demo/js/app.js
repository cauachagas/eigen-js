/**
 * Main Application Entrypoint - Drawer State, Global Navigation & Router Setup
 */

import { initRouter } from './router.js';

function initApp() {
  const appLayout = document.getElementById('appLayout');
  const navDrawer = document.getElementById('navDrawer');
  const menuToggle = document.getElementById('menuToggle');
  const drawerBackdrop = document.getElementById('drawerBackdrop');
  const viewContainer = document.getElementById('viewContainer');
  const currentRouteTitle = document.getElementById('currentRouteTitle');

  // Drawer toggle logic
  function toggleDrawer() {
    const isMobile = window.innerWidth <= 960;
    if (isMobile) {
      navDrawer.classList.toggle('open');
      drawerBackdrop.classList.toggle('active');
    } else {
      appLayout.classList.toggle('drawer-collapsed');
    }
  }

  function closeMobileDrawer() {
    navDrawer.classList.remove('open');
    drawerBackdrop.classList.remove('active');
  }

  if (menuToggle) {
    menuToggle.addEventListener('click', toggleDrawer);
  }

  if (drawerBackdrop) {
    drawerBackdrop.addEventListener('click', closeMobileDrawer);
  }

  // Close drawer on mobile when clicking a drawer item
  document.querySelectorAll('.drawer-item').forEach(link => {
    link.addEventListener('click', () => {
      if (window.innerWidth <= 960) {
        closeMobileDrawer();
      }
    });
  });

  // Handle escape key to close drawer
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') {
      closeMobileDrawer();
    }
  });

  // Initialize Router
  initRouter(viewContainer, (routeKey, route) => {
    if (currentRouteTitle) {
      currentRouteTitle.textContent = route.title === 'Eigen JS' ? '' : `/ ${route.title}`;
    }
  });

  // Check engines
  if (window.eig?.ready) {
    window.eig.ready.catch(err => console.error('Eigen.js failed to load:', err));
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}
