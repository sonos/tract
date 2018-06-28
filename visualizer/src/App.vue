<template>
  <v-app>
    <v-toolbar color="indigo" dark fixed app>
      <v-toolbar-title class="mx-4">
        TFVisualizer
      </v-toolbar-title>

      <v-text-field
        prepend-icon="search"
        hide-details
        single-line
        class="mx-4"></v-text-field>

      <v-btn icon
        @click="$refs.input.open()"
        class="ml-4">
        <v-icon>open_in_browser</v-icon>
      </v-btn>
      <v-btn icon class="mr-4">
        <v-icon>settings</v-icon>
      </v-btn>
    </v-toolbar>

    <v-content>
      <graph-area v-if="graph" style="width: 100%; height: 100%;" :graph="graph"></graph-area>

      <v-container v-else fill-height>
        <v-layout align-center>
          <v-flex>
            <h2 class="display-2">Welcome to TFVisualizer</h2>
            <span class="subheading">To get started, generate an analyser dump of your model by running <code>cli &lt;model&gt; --size &lt;size&gt; analyse</code> and load it.</span>
            <v-divider class="my-3"></v-divider>
            <v-btn
              large
              color="primary"
              class="mx-0"
              @click="$refs.input.open()">Load a dump</v-btn>
          </v-flex>
        </v-layout>
      </v-container>
    </v-content>

    <json-input
      v-model="graph"
      ref="input">
    </json-input>
  </v-app>
</template>

<script>
  import JsonInput from './JsonInput.vue'
  import GraphArea from './GraphArea.vue'

  export default {
    data: () => ({
      graph: null
    }),

    components: {
      JsonInput,
      GraphArea,
    },

    props: [],

    /** Fetches the last graph from the sessionStorage. */
    mounted() {
      let graph = sessionStorage.getItem('tfd-state')
      if (graph) {
        this.graph = JSON.parse(graph)
      } else {
        fetch('/current').then(resp => {
          if (resp.ok) {
            resp.json().then(graph => {
              this.graph = graph
            })
          }
        })
      }
    },

    watch: {
      /** Stores the graph in the sessionStorage when updated. */
      graph(value) {
        console.log('Storing the graph in the sessionStorage.')
        sessionStorage.setItem('tfd-state', JSON.stringify(value))
      }
    }
  }
</script>