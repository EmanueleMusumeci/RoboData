$(document).ready(function() {
    const graphUrls = {
        "batch_a_QA01-graph": "assets/knowledge_graphs/batch_a_QA01.html",
        "batch_a_QA02-graph": "assets/knowledge_graphs/batch_a_QA02.html",
        "batch_a_QA03-graph": "assets/knowledge_graphs/batch_a_QA03.html",
        "batch_a_QA04-graph": "assets/knowledge_graphs/batch_a_QA04.html",
        "batch_a_QA05-graph": "assets/knowledge_graphs/batch_a_QA05.html",
        "batch_a_QA06-graph": "assets/knowledge_graphs/batch_a_QA06.html",
        "batch_a_QA07-graph": "assets/knowledge_graphs/batch_a_QA07.html",
        "batch_b_QB01-graph": "assets/knowledge_graphs/batch_b_QB01.html",
        "batch_b_QB02-graph": "assets/knowledge_graphs/batch_b_QB02.html",
        "batch_b_QB03-graph": "assets/knowledge_graphs/batch_b_QB03.html",
        "batch_d_QD01-graph": "assets/knowledge_graphs/batch_d_QD01.html",
        "batch_d_QD02-graph": "assets/knowledge_graphs/batch_d_QD02.html",
        "batch_d_QD03-graph": "assets/knowledge_graphs/batch_d_QD03.html",
        "batch_d_QD04-graph": "assets/knowledge_graphs/batch_d_QD04.html",
        "batch_d_QD06-graph": "assets/knowledge_graphs/batch_d_QD06.html",
        "batch_d_QD07-graph": "assets/knowledge_graphs/batch_d_QD07.html",
        "batch_d_QD08-graph": "assets/knowledge_graphs/batch_d_QD08.html"
    };

    const loadedGraphs = {};

    function getRefColor(className) {
        const tempElement = $(`<div class="${className}" style="display: none;"></div>`);
        $('body').append(tempElement);
        const color = tempElement.css('color');
        tempElement.remove();
        return color;
    }

    const refColors = {};
    for (let i = 1; i <= 10; i++) {
        refColors[`ref${i}`] = getRefColor(`ref${i}`);
    }

    $('.collapsible-prompt .button').on('click', function() {
        const content = $(this).next('.prompt-content');
        const graphContainer = content.find('.graph-container');
        const graphId = graphContainer.attr('id');

        if (content.is(':visible')) {
            // Now handled by index.html script
        } else {
            if (graphId && !loadedGraphs[graphId]) {
                const graphUrl = graphUrls[graphId];
                if (graphUrl) {
                    loadGraph(graphContainer, graphUrl, graphId);
                } else {
                    console.error("No URL found for graph ID:", graphId);
                }
            }
        }
    });

    function loadGraph(container, url, graphId) {
        $.ajax({
            url: url,
            dataType: "html",
            success: function(html) {
                const nodesMatch = html.match(/var nodes = new vis.DataSet\(([\s\S]*?)\);/);
                const edgesMatch = html.match(/var edges = new vis.DataSet\(([\s\S]*?)\);/);

                if (nodesMatch && edgesMatch) {
                    let nodes, edges;
                    try {
                        nodes = new vis.DataSet(eval(nodesMatch[1]));
                        edges = new vis.DataSet(eval(edgesMatch[1]));
                    } catch (e) {
                        console.error("Error parsing nodes/edges from", url, e);
                        container.html("Failed to load graph data.");
                        return;
                    }
                    
                    const rowId = container.closest('tr').prev('tr').prev('tr').attr('id');
                    colorizeGraphElements(nodes, edges, rowId);

                    const data = { nodes: nodes, edges: edges };
                    const options = {
                        nodes: {
                            shape: 'box',
                            font: {
                                size: 14,
                                face: 'arial'
                            }
                        },
                        edges: {
                            font: {
                                size: 12,
                                align: 'middle'
                            },
                            arrows: 'to'
                        },
                        physics: {
                            solver: 'repulsion',
                            repulsion: {
                                nodeDistance: 200
                            }
                        }
                    };

                    const network = new vis.Network(container[0], data, options);
                    network.on("stabilizationIterationsDone", function () {
                        network.fit();
                    });
                    
                    loadedGraphs[graphId] = { network, nodes, edges };
                    setupHoverInteractions(rowId, network, nodes, edges);

                } else {
                    container.html("Could not parse graph data from the source file.");
                    console.error("Could not find nodes/edges data in", url);
                }
            },
            error: function(xhr, status, error) {
                container.html("Knowledge graph not available.");
                console.error("Failed to load graph HTML:", error);
            }
        });
    }

    function colorizeGraphElements(nodes, edges, rowId) {
        if (!rowId) return;

        const supportSets = $(`#${rowId} .ref, #${rowId} + tr .ref`);
        supportSets.each(function() {
            const refClass = $(this).attr('class').split(' ').find(c => c.startsWith('ref'));
            if (!refClass) return;

            const color = refColors[refClass];
            const supportText = $(this).text();
            const nodeIds = supportText.match(/Q\d+/g);

            if (nodeIds && nodeIds.length > 0) {
                nodeIds.forEach(id => {
                    if (nodes.get(id)) {
                        nodes.update({ id: id, color: { background: color, border: color }, font: {color: 'white'} });
                    }
                    // Color edges connected to this node
                    const connectedEdges = edges.get({
                        filter: function(edge) {
                            return edge.from === id || edge.to === id;
                        }
                    });
                    connectedEdges.forEach(edge => {
                        edges.update({ id: edge.id, color: color });
                    });
                });
            }
        });
    }

    function setupHoverInteractions(rowId, network, nodes, edges) {
        const originalNodeColors = {};
        nodes.forEach(node => {
            originalNodeColors[node.id] = { background: node.color?.background, border: node.color?.border, font: node.font?.color };
        });

        const originalEdgeColors = {};
        edges.forEach(edge => {
            originalEdgeColors[edge.id] = edge.color;
        });

        const highlightColor = '#FFFF00'; // Yellow

        $(`#${rowId} .ref, #${rowId} + tr .ref`).hover(
            function() { // MOUSE IN
                const supportText = $(this).text();
                const nodeIds = supportText.match(/Q\d+/g) || [];
                
                // Highlight table cell and answer text
                $(this).css('background-color', highlightColor);

                // Highlight graph nodes and edges
                nodeIds.forEach(id => {
                    if (nodes.get(id)) {
                        nodes.update({ id: id, color: { background: highlightColor, border: 'black' }, font: {color: 'black'} });
                    }
                    const connectedEdges = edges.get({
                        filter: edge => edge.from === id || edge.to === id
                    });
                    connectedEdges.forEach(edge => {
                        edges.update({ id: edge.id, color: highlightColor, width: 3 });
                    });
                });
            },
            function() { // MOUSE OUT
                const supportText = $(this).text();
                const nodeIds = supportText.match(/Q\d+/g) || [];

                // Un-highlight table cell and answer text
                $(this).css('background-color', '');

                // Restore original graph colors
                nodeIds.forEach(id => {
                    if (nodes.get(id)) {
                        const original = originalNodeColors[id];
                        nodes.update({ id: id, color: { background: original.background, border: original.border }, font: {color: original.font} });
                    }
                    const connectedEdges = edges.get({
                        filter: edge => edge.from === id || edge.to === id
                    });
                    connectedEdges.forEach(edge => {
                        edges.update({ id: edge.id, color: originalEdgeColors[edge.id], width: 1 });
                    });
                });
            }
        );
    }
});
