/** A custom Cytoscape layout for Tensorflow graphs. */
class TensorflowLayout {
  constructor(options) {
    this.options = options
  }

  run() {
    console.log('Running layout!')
  }
}

// Export the layout as a Cytoscape extension.
export default (cytoscape) => {
  cytoscape('layout', 'tensorflow', TensorflowLayout)
}