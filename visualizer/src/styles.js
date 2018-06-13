export const graphStyle = [
  {
    selector: 'edge, node',
    style: {
      'font-size': '9px',
      'label': 'data(label)',
      'text-wrap': 'wrap',
      'font-family': 'Space Mono, monospace'
    }
  },

  {
    selector: 'node',
    style: {
      'border-width': '1px',
    }
  },

  {
    selector: 'node[background]',
    style: {'background-color': 'data(background)'}
  },

  {
    selector: 'node[border]',
    style: {'border-color': 'data(border)'}
  },

  {
    selector: 'node.highlighted',
    style: {
      'border-style': 'dotted',
      'border-color': '#666',
    }
  },

  {
    selector: 'node[op != "Const"]',
    style: {
      'shape': 'roundrectangle',
      'font-size': '11px',
      'width': 'label',
      'padding': '12px',
      'height': '8px',
      'text-valign': 'center',
    }
  },

  {
    selector: 'node[op = "Const"]',
    style: {
      'shape': 'ellipse',
      'width': '10px',
      'height': '10px',
      'text-margin-y': '-5px',
      'background-color': '#ddd',
      'border-color': '#bbb',
    }
  },

  {
    selector: 'node[op = "Meta"]',
    style: {
      'font-size': '9px',
      'shape': 'roundrectangle',
      'padding': '10px',
      'background-color': '#eee',
      'border-color': '#ddd',
      'border-style': 'dashed',
      'text-rotation': '-90deg',
      'text-valign': 'top',
      'text-halign': 'left',
      'text-margin-x': '-7px',
      'text-margin-y': '2px',
      'color': '#888',
    }
  },

  {
    selector: 'node[op = "Meta"].cy-expand-collapse-collapsed-node',
    style: {
      'font-size': '11px',
      'padding': '12px',
      'height': '8px',
      'width': 'label',
      'text-rotation': '0deg',
      'text-valign': 'center',
      'text-halign': 'center',
      'text-margin-x': '0',
      'text-margin-y': '0',
      'color': '#000',
    }
  },

  {
    selector: 'edge',
    style: {
      'width': '1px',
      'line-color': '#ccc',
      'curve-style': 'bezier',
      'target-arrow-color': '#ccc',
      'target-arrow-shape': 'vee',
      'arrow-scale': '.8',
      'color': '#999',
      'font-size': '8px',
      'text-background-color': '#fff',
      'text-background-opacity': '.5',
      'text-background-padding': '2px',
      'text-background-shape': 'roundrectangle',
    }
  },

  {
    selector: 'edge[?constant]',
    style: {
      'line-style': 'dashed',
      'label': '',
    }
  },

  {
    selector: 'edge.highlighted',
    style: {
      'line-style': 'dotted',
      'line-color': '#666',
      'target-arrow-color': '#666',
    }
  },
]