/* eslint-disable */
(function () {
  function $(id) { return document.getElementById(id); }

  function setText(id, text, fallback) {
    var el = $(id);
    if (!el) return;
    el.textContent = (text === undefined || text === null || text === '') ? (fallback || '') : String(text);
  }

  function setHTML(id, html) {
    var el = $(id);
    if (!el) return;
    el.innerHTML = html || '';
  }

  function createEl(tag, className, html) {
    var el = document.createElement(tag);
    if (className) el.className = className;
    if (html !== undefined) el.innerHTML = html;
    return el;
  }

  function safeArray(val) {
    return Array.isArray(val) ? val : [];
  }

  function get(data, path, fallback) {
    try {
      return path.split('.').reduce(function (acc, key) { return acc && acc[key] !== undefined ? acc[key] : undefined; }, data) ?? fallback;
    } catch (e) {
      return fallback;
    }
  }

  function renderStats(data) {
    var gaps = safeArray(get(data, 'all_agent_data.gap_finder_act.identified_market_gaps')); 
    var recs = safeArray(get(data, 'all_agent_data.gap_finder_act.strategic_recommendations'));
    var painValidated = safeArray(get(data, 'all_agent_data.validator_act.validated_pain_points'));
    var painCount = painValidated.length || (get(data, 'all_agent_data.scout_act.total_discovered') || 0);
    var sources = safeArray(get(data, 'all_agent_data.scout_act.sources_used'));

    setText('stat-gaps', gaps.length);
    setText('stat-recs', recs.length);
    setText('stat-pain', painCount);
    setText('stat-sources', sources.length);
  }

  function renderExecutive(data) {
    setText('executive-summary', get(data, 'think_analysis.report_analysis.executive_summary', ''));
    setText('primary-opportunity', get(data, 'think_analysis.report_analysis.primary_business_opportunity', ''));

    var keyMetrics = safeArray(get(data, 'think_analysis.content_planning.key_metrics_to_highlight'));
    var metricsWrap = $('key-metrics');
    if (metricsWrap) {
      metricsWrap.innerHTML = '';
      keyMetrics.forEach(function (m) {
        var card = createEl('div', 'card');
        card.textContent = m;
        metricsWrap.appendChild(card);
      });
    }
  }

  function renderPainPoints(data) {
    var list = $('pain-points');
    if (!list) return;
    list.innerHTML = '';
    var pains = safeArray(get(data, 'all_agent_data.scout_act.pain_points'));
    var validated = safeArray(get(data, 'all_agent_data.validator_act.validated_pain_points'));

    var idToValidation = {};
    validated.forEach(function (v, idx) { idToValidation[v.description] = v.validation_status || 'validated'; });

    pains.forEach(function (p) {
      var card = createEl('div', 'card');
      var title = createEl('h3', null, p.description || 'Pain point');
      var meta = createEl('div', 'muted');
      var status = idToValidation[p.description] ? '<span class="badge">' + idToValidation[p.description] + '</span>' : '';
      var sev = p.severity ? '<span class="badge ' + (p.severity === 'high' ? 'high' : 'medium') + '">' + p.severity + '</span>' : '';
      var score = (p.impact_score !== undefined) ? ('Impact: ' + p.impact_score) : '';
      meta.innerHTML = [status, sev, score].filter(Boolean).join(' ');
      card.appendChild(title);
      card.appendChild(meta);
      if (p.evidence && p.evidence.length) {
        var ul = createEl('ul', 'list');
        p.evidence.forEach(function (e) { ul.appendChild(createEl('li', null, e)); });
        card.appendChild(ul);
      }
      list.appendChild(card);
    });
  }

  function renderGaps(data) {
    var wrap = $('market-gaps');
    if (!wrap) return;
    wrap.innerHTML = '';
    var gaps = safeArray(get(data, 'all_agent_data.gap_finder_act.identified_market_gaps'));
    gaps.forEach(function (g) {
      var card = createEl('div', 'card');
      card.appendChild(createEl('h3', null, (g.gap_name || g.title || '').replace(/^\*\*|\*\*$/g, '')));
      if (g.description) card.appendChild(createEl('p', 'muted', g.description));
      var sev = g.severity || '';
      if (sev) card.appendChild(createEl('div', null, '<span class="badge ' + (sev.toLowerCase().includes('high') ? 'high' : 'medium') + '">' + sev + '</span>'));
      wrap.appendChild(card);
    });
  }

  function renderRecommendations(data) {
    var wrap = $('recommendations');
    if (!wrap) return;
    wrap.innerHTML = '';
    var recs = safeArray(get(data, 'all_agent_data.gap_finder_act.strategic_recommendations'));
    recs.forEach(function (r) {
      var card = createEl('div', 'card');
      card.appendChild(createEl('h3', null, (r.recommendation_name || '').replace(/^\*\*|\*\*$/g, '')));
      if (r.approach) card.appendChild(createEl('p', 'muted', r.approach));
      if (r.target_market) card.appendChild(createEl('p', 'muted', '<span class="badge">Target</span> ' + r.target_market));
      wrap.appendChild(card);
    });
  }

  function renderSolution(data) {
    setText('solution-name', get(data, 'all_agent_data.builder_act.business_solution_summary.solution_name', 'N/A'));
    setText('solution-concept', get(data, 'all_agent_data.builder_act.business_solution_summary.business_concept', ''));
    setText('target-market', get(data, 'all_agent_data.builder_act.business_solution_summary.target_market', ''));

    var core = safeArray(get(data, 'all_agent_data.builder_act.product_strategy.core_features'));
    var adv = safeArray(get(data, 'all_agent_data.builder_act.product_strategy.advanced_features'));
    var coreUl = $('core-features');
    var advUl = $('advanced-features');
    if (coreUl) { coreUl.innerHTML = ''; core.forEach(function (f) { coreUl.appendChild(createEl('li', null, f)); }); }
    if (advUl) { advUl.innerHTML = ''; adv.forEach(function (f) { advUl.appendChild(createEl('li', null, f)); }); }

    // Pricing & economics
    var pricingList = $('pricing-strategy');
    var pricing = get(data, 'all_agent_data.builder_act.business_model_pricing.pricing_strategy', '');
    if (pricingList) {
      pricingList.innerHTML = '';
      (String(pricing).split(',') || []).map(function (s) { return s.trim(); }).filter(Boolean).forEach(function (item) {
        pricingList.appendChild(createEl('li', null, item));
      });
    }
    setText('unit-economics', get(data, 'all_agent_data.builder_act.business_model_pricing.unit_economics', ''));
    setText('profitability', get(data, 'all_agent_data.builder_act.business_model_pricing.profitability_timeline', ''));
  }

  function renderKpis(data) {
    var ul = $('kpis');
    if (ul) {
      ul.innerHTML = '';
      safeArray(get(data, 'all_agent_data.builder_act.success_metrics_milestones.key_performance_indicators')).forEach(function (k) {
        ul.appendChild(createEl('li', null, k));
      });
    }
    setText('milestones', get(data, 'all_agent_data.builder_act.success_metrics_milestones.milestone_timeline', ''));
  }

  function renderRisks(data) {
    var wrap = $('risks');
    if (!wrap) return;
    wrap.innerHTML = '';
    var risks = safeArray(get(data, 'all_agent_data.builder_act.success_metrics_milestones.risk_mitigation'));
    if (!risks.length) {
      risks = safeArray(get(data, 'all_agent_data.gap_finder_think.risks_and_unknowns'));
    }
    risks.forEach(function (r) {
      var card = createEl('div', 'card');
      var title = r.description || r.risk || 'Risk';
      card.appendChild(createEl('h3', null, title));
      var meta = createEl('div', 'muted');
      var impact = r.impact ? '<span class="badge ' + (String(r.impact).toLowerCase().startsWith('h') ? 'high' : 'medium') + '">impact: ' + r.impact + '</span>' : '';
      var likelihood = r.likelihood ? '<span class="badge">likelihood: ' + r.likelihood + '</span>' : '';
      meta.innerHTML = [impact, likelihood].filter(Boolean).join(' ');
      card.appendChild(meta);
      if (r.mitigation_strategy) card.appendChild(createEl('p', 'muted', r.mitigation_strategy));
      wrap.appendChild(card);
    });
  }

  function renderMethodology(data) {
    var sources = safeArray(get(data, 'all_agent_data.scout_act.sources_used'));
    var ul = $('sources');
    if (ul) { ul.innerHTML = ''; sources.forEach(function (s) { ul.appendChild(createEl('li', null, s)); }); }

    var conf = get(data, 'all_agent_data.scout_act.confidence_score');
    var thinkConf = get(data, 'all_agent_data.gap_finder_think.summary.analysis_confidence');
    setText('confidence', (conf !== undefined ? ('Scout confidence: ' + conf) : '') + (thinkConf ? (' | Analysis: ' + thinkConf) : ''));
  }

  function hydrate(data) {
    renderStats(data);
    renderExecutive(data);
    renderPainPoints(data);
    renderGaps(data);
    renderRecommendations(data);
    renderSolution(data);
    renderKpis(data);
    renderRisks(data);
    renderMethodology(data);
  }

  function tryParseEmbedded() {
    var tag = document.getElementById('analysis-data');
    if (!tag) return null;
    var raw = tag.textContent && tag.textContent.trim();
    if (!raw) return null;
    try { return JSON.parse(raw); } catch (e) { return null; }
  }

  function init() {
    var embedded = tryParseEmbedded();
    if (embedded) {
      hydrate(embedded);
      return;
    }
    // Fallback: fetch the JSON file in the same folder
    fetch('analysis_data_20250928_154103.json', { cache: 'no-store' })
      .then(function (res) { return res.json(); })
      .then(function (json) { hydrate(json); })
      .catch(function () {
        setHTML('executive-summary', '<em>Unable to load analysis data.</em>');
      });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();


