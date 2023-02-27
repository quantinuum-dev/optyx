use std::collections::{HashMap, HashSet};

use num::Rational;
use quizx;

use crate::graph::*;


/// Different vertex types for ZX diagrams
#[derive(Debug,Copy,Clone,PartialEq,Eq,PartialOrd,Ord)]
pub enum ZX {
    Z(Rational),
    X(Rational),
    H,
    Inp,
    Out
}

impl VType for ZX {
    fn num_expected_inputs(&self) -> Option<usize> {
        match self {
            ZX::Z(_) => None,
            ZX::X(_) => None,
            ZX::H => Some(1),
            ZX::Inp => None,
            ZX::Out => None
        }
    }

    fn num_expected_outputs(&self) -> Option<usize> {
        match self {
            ZX::Z(_) => None,
            ZX::X(_) => None,
            ZX::H => Some(1),
            ZX::Inp => None,
            ZX::Out => None
        }
    }
}

/// Our ZX diagrams do not have edge labels (we encode Hadamards as vertices to make the translation
/// to QPath easier).
type ZXEdge = ();
impl EType for ZXEdge {}


/// The type of ZX graphs
pub type ZXGraph = VecGraph<ZX, ZXEdge>;

impl ZXGraph {
    /// Converts a quizx ZX graph into our representation. In particular, we direct the edges of
    /// the diagram to get a DAG.
    pub fn from_quizx<G: quizx::graph::GraphLike>(quizx_graph: &G) -> Self {
        let mut g = ZXGraph::new();
        *g.scalar_mut() *= quizx_graph.scalar();

        // Add vertices
        let mut map: HashMap<quizx::graph::V, V> = HashMap::new();
        for v in quizx_graph.vertices() {
            match quizx_graph.vertex_type(v) {
                quizx::graph::VType::Z => map.insert(v, g.add_vertex(ZX::Z(quizx_graph.phase(v))).0),
                quizx::graph::VType::X => map.insert(v, g.add_vertex(ZX::X(quizx_graph.phase(v))).0),
                quizx::graph::VType::B => continue,  // Skip boundary vertices to differentiate inputs and outputs
                quizx::graph::VType::H => panic!("H boxes not supported")
            };
        }

        // Create input and output vertices
        for &v in quizx_graph.inputs() {
            map.insert(v, g.add_vertex(ZX::Inp).0);
        }
        for &v in quizx_graph.outputs() {
            map.insert(v, g.add_vertex(ZX::Out).0);
        }

        let inputs: HashSet<quizx::graph::V> = HashSet::from_iter(quizx_graph.inputs().iter().map(|&x| x));
        let outputs: HashSet<quizx::graph::V> = HashSet::from_iter(quizx_graph.outputs().iter().map(|&x| x));

        for (u, v, ety) in quizx_graph.edges() {
            let s: quizx::graph::V;
            let t: quizx::graph::V;
            if inputs.contains(&u) || outputs.contains(&v) {
                s = u;
                t = v;
            } else if outputs.contains(&u) || inputs.contains(&v) {
                s = v;
                t = u;
            } else {
                s = std::cmp::min(u, v);
                t = std::cmp::max(u, v);
            }
            g.add_edge_with_type(map[&s], map[&t], ety)
        }
         
        g
    }

    fn add_edge_with_type(&mut self, u: V, v: V, ety: quizx::graph::EType) {
        let u_port = self.add_out_port(u);
        let v_port = self.add_in_port(v);

        match ety {
            quizx::vec_graph::EType::N => self.add_edge(u_port, v_port, ()),
            quizx::vec_graph::EType::H => {
                let (_, h_in, h_out) = self.add_vertex(ZX::H);
                self.add_edge(u_port, h_in[0], ());
                self.add_edge(h_out[0], v_port, ());
            }
        }
    }

}

