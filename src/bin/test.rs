use num::Rational;
use quizx::circuit::*;
use quizx::vec_graph::GraphLike;

use optyx::graph::Graph;
use optyx::zx::ZXGraph;
use optyx::qpath::QPathGraph;


fn main() {
    let mut circ = Circuit::new(3);
    circ.add_gate("cx", vec![0, 1]);
    circ.add_gate("h", vec![2]);
    circ.add_gate_with_phase("rz", vec![0], Rational::new(1, 3));
    circ.add_gate("cx", vec![1, 2]);
    circ.add_gate_with_phase("rz", vec![1], Rational::new(1, 4));
    circ.add_gate("cx", vec![0, 1]);

    let mut g: quizx::hash_graph::Graph = circ.to_graph();
    g.x_to_z();

    let g = ZXGraph::from_quizx(&g);

    let g = QPathGraph::from_zx(&g);
    
    println!("{}", g.to_dot());
}
