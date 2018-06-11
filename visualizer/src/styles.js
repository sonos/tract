export const graphStyle = [
  {
    selector: 'node',
    style: {
      'font-size': '9px',
      'border-width': '1px',
      'label': 'data(name)',
      'font-family': 'Space Mono, monospace'
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
    selector: 'node[op != "Const"]',
    style: {
      'shape': 'roundrectangle',
      'font-size': '11px',
      'width': 'label',
      'padding': '8px',
      'height': '15px',
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
    selector: 'node[type = "metanode"]',
    style: {
      'shape': 'roundrectangle',
      'padding': '8px',
      'background-color': '#eee',
      'border-color': '#ddd',
      'border-style': 'dashed',
      'text-rotation': '-90deg',
      'text-valign': 'top',
      'text-halign': 'left',
      'text-margin-x': '-6px',
      'text-margin-y': '2px',
      'color': '#888',
    }
  },

  {
    selector: 'node[type = "metanode"].cy-expand-collapse-collapsed-node',
    style: {
      'font-size': '11px',
      'height': '15px',
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
      'width': 1,
      'line-color': '#ccc',
      'curve-style': 'bezier',
      'target-arrow-color': '#ccc',
      'target-arrow-shape': 'vee',
      'arrow-scale': '.8',
    }
  }
]