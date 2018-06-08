export const graphStyle = [
  {
    selector: 'node',
    style: {
      'background-color': '#ddd',
      'border-width': '1px',
      'border-color': '#bbb',
      'shape': 'ellipse',
      'label': 'data(name)',
      'font-size': '9px',
      'text-margin-y': '-8px',
      'width': '25px',
      'height': '15px',
    }
  },

  {
    selector: 'node[op = "Const"]',
    style: {
      'width': '10px',
      'height': '10px',
    }
  },

  {
    selector: '$node > node',
    css: {
      'padding-top': '10px',
      'padding-left': '10px',
      'padding-bottom': '10px',
      'padding-right': '10px',
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