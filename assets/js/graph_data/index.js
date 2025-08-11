// Index of all available graph data files
// Generated automatically

window.availableGraphs = [
  "batch_a_QA01",
  "batch_a_QA02",
  "batch_a_QA03",
  "batch_a_QA04",
  "batch_a_QA05",
  "batch_a_QA06",
  "batch_a_QA07",
  "batch_b_QB01",
  "batch_b_QB02",
  "batch_b_QB03",
  "batch_d_QD01",
  "batch_d_QD02",
  "batch_d_QD03",
  "batch_d_QD04",
  "batch_d_QD06",
  "batch_d_QD07",
  "batch_d_QD08"
];

// Function to load graph data
window.loadGraphData = function(graphId) {
    return window.graphData && window.graphData[graphId] ? window.graphData[graphId] : null;
};
