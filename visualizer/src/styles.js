export const graphStyle = [
  {
    selector: 'node',
    style: {
      'background-color': '#ddd',
      'border-width': '1px',
      'border-color': '#bbb',
      'label': 'data(name)',
    }
  },

  {
    selector: 'node[op != "Const"]',
    style: {
      'shape': 'roundrectangle',
      'font-size': '9px',
      'width': 'label',
      'padding': '5px',
      'height': '15px',
      'text-valign': 'center',
    }
  },

  {
    selector: 'node[op = "Const"]',
    style: {
      'shape': 'ellipse',
      'font-size': '8px',
      'width': '10px',
      'height': '10px',
      'text-margin-y': '-6px',
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