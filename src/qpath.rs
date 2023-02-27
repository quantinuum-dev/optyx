use std::collections::HashMap;
use num::Rational;
use num::Zero;
use crate::graph::*;
use crate::zx::*;


#[derive(Debug,Copy,Clone,PartialEq,Eq,PartialOrd,Ord)]
pub enum QP {
    Input,
    Output,
    Monoid,
    Comonoid,
    Unit,
    Counit,
    Create,
    Annil,
    TBS(Rational),  // Tunable Beam Splitter
    BBS(Rational),  // Biases Beam Splitter
}

impl VType for QP {
    fn num_expected_inputs(&self) -> Option<usize> {
        Some (match self {
            QP::Input => 0,
            QP::Output => 1,
            QP::Monoid => 1,
            QP::Comonoid => 2,
            QP::Unit => 0,
            QP::Counit => 1,
            QP::Create => 0,
            QP::Annil => 1,
            QP::TBS(_) => 2,
            QP::BBS(_) => 2
        })
    }

    fn num_expected_outputs(&self) -> Option<usize> {
        Some (match self {
            QP::Input => 1,
            QP::Output => 0,
            QP::Monoid => 2,
            QP::Comonoid => 1,
            QP::Unit => 1,
            QP::Counit => 0,
            QP::Create => 1,
            QP::Annil => 0,
            QP::TBS(_) => 2,
            QP::BBS(_) => 2
        })
    }
}

impl EType for Rational { }

pub type QPathGraph = VecGraph<QP, Rational>;


/// We define an extended version of QPath with Id and Phase nodes
/// to make the wiring easier. We remove them later in an extra step
#[derive(Debug,Copy,Clone,PartialEq,Eq,PartialOrd,Ord)]
enum ExtQP {
    QP(QP),
    Id,
    Phase(Rational)
}

impl VType for ExtQP {
    fn num_expected_inputs(&self) -> Option<usize> {
        match self {
            ExtQP::QP(v) => v.num_expected_inputs(),
            ExtQP::Id => Some(1),
            ExtQP::Phase(_) => Some(1)
        }
    }

    fn num_expected_outputs(&self) -> Option<usize> {
        match self {
            ExtQP::QP(v) => v.num_expected_outputs(),
            ExtQP::Id => Some(1),
            ExtQP::Phase(_) => Some(1)
        }
    }
}

type ExtQPathGraph = VecGraph<ExtQP, ()>;

impl ExtQPathGraph {
    pub fn from_zx(zx: &ZXGraph) -> Self {
        let mut g = ExtQPathGraph::new();

        // Mapping of nodes in ZX to dual rail encoding in g
        let mut in_ports: HashMap<InPort, (InPort, InPort)> = HashMap::new();
        let mut out_ports: HashMap<OutPort, (OutPort, OutPort)> = HashMap::new();

        for v in zx.vertices() {
            let (inp, out) = match zx.vertex_type(v) {
                ZX::H => g.add_had(),
                ZX::Z(phase) => g.add_zspider(zx.incident_edges_in(v).len(), zx.incident_edges_out(v).len(), phase),
                ZX::Inp => g.add_input(),
                ZX::Out => g.add_output(),
                _ => panic!("ZX Vertex type not supported")
            };
            for (i, &port) in zx.in_ports(v).iter().enumerate() {
                in_ports.insert(port, inp[i]);
            }
            for (i, &port) in zx.out_ports(v).iter().enumerate() {
                out_ports.insert(port, out[i]);
            }
        }

        // Wire up vertices
        for (u, v, ()) in zx.edges() {
            let (start1, start2) = out_ports[&u];
            let (end1, end2) = in_ports[&v];

            g.add_edge(start1, end1, ());
            g.add_edge(start2, end2, ());
        }

        g
    }

    fn add_monoid(&mut self) -> (InPort, (OutPort, OutPort)) {
        let (_, inp, out) = self.add_vertex(ExtQP::QP(QP::Monoid));
        (inp[0], (out[0], out[1]))
    }

    fn add_comonoid(&mut self) -> ((InPort, InPort), OutPort) {
        let (_, inp, out) = self.add_vertex(ExtQP::QP(QP::Comonoid));
        ((inp[0], inp[1]), out[0])
    }

    fn add_unit(&mut self) -> OutPort {
        let (_, _, out) = self.add_vertex(ExtQP::QP(QP::Unit));
        out[0]
    }

    fn add_counit(&mut self) -> InPort {
        let (_, inp, _) = self.add_vertex(ExtQP::QP(QP::Counit));
        inp[0]
    }

    fn add_create(&mut self) -> OutPort {
        let (_, _, out) = self.add_vertex(ExtQP::QP(QP::Create));
        out[0]
    }

    fn add_annil(&mut self) -> InPort {
        let (_, inp, _) = self.add_vertex(ExtQP::QP(QP::Annil));
        inp[0]
    }

    fn add_tbs(&mut self, phase: Rational) -> ((InPort, InPort), (OutPort, OutPort)) {
        let (_, inp, out) = self.add_vertex(ExtQP::QP(QP::TBS(phase)));
        ((inp[0], inp[1]), (out[0], out[1]))
    }

    fn add_bbs(&mut self, phase: Rational) -> ((InPort, InPort), (OutPort, OutPort)) {
        let (_, inp, out) = self.add_vertex(ExtQP::QP(QP::BBS(phase)));
        ((inp[0], inp[1]), (out[0], out[1]))
    }

    fn add_id(&mut self) -> (InPort, OutPort) {
        let (_, inp, out) = self.add_vertex(ExtQP::Id);
        (inp[0], out[0])
    }

    fn add_phase(&mut self, phase: Rational) -> (InPort, OutPort) {
        let (_, inp, out) = self.add_vertex(ExtQP::Phase(phase));
        (inp[0], out[0])
    }

    // Add a black dot made up of a Create and Monoids
    fn add_split(&mut self) -> (OutPort, OutPort) {
        let create = self.add_create();
        let (mon_in, mon_outs) = self.add_monoid();
        self.add_edge(create, mon_in, ());
        mon_outs
    }

    // Add a black dot made up of a Comonoid and Annil
    fn add_fuse(&mut self) -> (InPort, InPort) {
        let (mon_ins, mon_out) = self.add_comonoid();
        let annil = self.add_annil();
        self.add_edge(mon_out, annil, ());
        mon_ins
    }


    fn add_input(&mut self) -> (Vec<(InPort, InPort)>, Vec<(OutPort, OutPort)>) {
        let (_, _, outs1) = self.add_vertex(ExtQP::QP(QP::Input));
        let (_, _, outs2) = self.add_vertex(ExtQP::QP(QP::Input));
        (vec![], vec![(outs1[0], outs2[0])])
    }

    fn add_output(&mut self) -> (Vec<(InPort, InPort)>, Vec<(OutPort, OutPort)>) {
        let (_, inp1, _) = self.add_vertex(ExtQP::QP(QP::Output));
        let (_, inp2, _) = self.add_vertex(ExtQP::QP(QP::Output));
        (vec![(inp1[0], inp2[0])], vec![])
    }

    fn add_had(&mut self) -> (Vec<(InPort, InPort)>, Vec<(OutPort, OutPort)>) {
        let (ins, outs) = self.add_tbs(Rational::new(1, 4));
        (vec![ins], vec![outs])
    }

    fn add_zspider(&mut self, num_inputs: usize, num_outputs: usize, phase: Rational) -> (Vec<(InPort, InPort)>, Vec<(OutPort, OutPort)>) {
        let mut mapping = match (num_inputs, num_outputs) {
            // Scalar
            (0, 0) => {
                return (vec![], vec![]);
            }
            // Z spider state
            (0, 1) => {
                let outs = self.add_split();
                (vec![], vec![outs])
            }
            // Z spider effect
            (1, 0) => {
                let ins = self.add_fuse();
                (vec![ins], vec![])
            }
            // Identity
            (1, 1) => {
                let (id1_in, id1_out) = self.add_id();
                let (id2_in, id2_out) = self.add_id();
                (vec![(id1_in, id2_in)], vec![(id1_out, id2_out)])
            }
            // Z Fusion
            (2, 1) => {
                let (id1_in, id1_out) = self.add_id();
                let (fusion1, fusion2) = self.add_fuse();
                let (id2_in, id2_out) = self.add_id();
                (vec![(id1_in, fusion1), (fusion2, id2_in)], vec![(id1_out, id2_out)])
            }
            // Z Copy
            (1, 2) => {
                let (split1_out1, split1_out2) = self.add_split();
                let (split2_out1, split2_out2) = self.add_split();
                let (split3_out1, split3_out2) = self.add_split();
                let ((bbs_in1, bbs_in2), (bbs_out1, bbs_out2)) = self.add_bbs(Rational::new(1, 2));
                let ((bs_in1, bs_in2), (bs_out1, bs_out2)) = self.add_bbs(Rational::zero());
                let (fuse1_in1, fuse1_in2) = self.add_fuse();
                let (fuse2_in1, fuse2_in2) = self.add_fuse();
                let (id_in, id_out) = self.add_id();
                self.add_edge(split1_out2, bbs_in2, ());
                self.add_edge(split2_out2, bbs_in1, ());
                self.add_edge(split3_out1, bs_in2, ());
                self.add_edge(bbs_out1, fuse1_in1, ());
                self.add_edge(bbs_out2, fuse2_in1, ());
                self.add_edge(bs_out1, fuse2_in2, ());
                self.add_edge(bs_out2, fuse1_in2, ());
                (vec![(bs_in1, id_in)], vec![(split1_out1, split2_out1), (split3_out2, id_out)])
            }
            // Generalised Z Fusion
            (n, 1) => {
                let mut fuse_rec = self.add_zspider(n-1, 1, Rational::zero());
                let mut fuse_base = self.add_zspider(2, 1, Rational::zero());
                let (in1, in2) = fuse_base.0.pop().unwrap();
                let (out1, out2) = fuse_rec.1.pop().unwrap();
                self.add_edge(out1, in1, ());
                self.add_edge(out2, in2, ());
                fuse_rec.0.append(&mut fuse_base.0);
                fuse_rec.1.append(&mut fuse_base.1);
                fuse_rec
            }
            // Generalised Z Copy
            (1, m) => {
                let mut copy_rec = self.add_zspider(1, m-1, Rational::zero());
                let mut copy_base = self.add_zspider(1, 2, Rational::zero());
                let (in1, in2) = copy_base.0.pop().unwrap();
                let (out1, out2) = copy_rec.1.pop().unwrap();
                self.add_edge(out1, in1, ());
                self.add_edge(out2, in2, ());
                copy_rec.0.append(&mut copy_base.0);
                copy_rec.1.append(&mut copy_base.1);
                copy_rec
            }
            // Represent any other spider as combination of fusion and copy
            (n, m) => {
                let mut fusion = self.add_zspider(n, 1, Rational::zero());
                let mut copy = self.add_zspider(1, m, Rational::zero());
                let (in1, in2) = copy.0.pop().unwrap();
                let (out1, out2) = fusion.1.pop().unwrap();
                self.add_edge(out1, in1, ());
                self.add_edge(out2, in2, ());
                fusion.0.append(&mut copy.0);
                fusion.1.append(&mut copy.1);
                fusion
            }
        };

        // Add phase to arbitrary leg
        if !phase.is_zero() {
            let (ph_in, ph_out) = self.add_phase(phase);
            if num_inputs > 0 {
                let (in1, in2) = mapping.0.pop().unwrap();
                self.add_edge(ph_out, in2, ());
                mapping.0.push((in1, ph_in));
            } else {
                let (out1, out2) = mapping.1.pop().unwrap();
                self.add_edge(out2, ph_in, ());
                mapping.1.push((out1, ph_out));
            }
        }

        mapping
    }

}


impl QPathGraph {
    pub fn from_zx(zx: &ZXGraph) -> Self {
        let mut g = ExtQPathGraph::from_zx(&zx).map_edges(Box::new(|()| Rational::zero())) as VecGraph<ExtQP, Rational>;
        
        for v in Vec::from_iter(g.vertices()) {
            match g.vertex_type(v) {
                ExtQP::Id => {
                    let a = g.incident_edges_in(v)[0].0;
                    let b = g.incident_edges_out(v)[0].1;
                    g.remove_vertex(v);
                    g.add_edge(a, b, Rational::zero());
                }
                ExtQP::Phase(phase) => {
                    let a = g.incident_edges_in(v)[0].0;
                    let b = g.incident_edges_out(v)[0].1;
                    g.remove_vertex(v);
                    g.add_edge(a, b, phase);
                }
                ExtQP::QP(_) => {}
            }
        }

        g.map_verts(Box::new(|v| match v {
            ExtQP::QP(x) => x,
            _ => panic!("Auxilliary vertex not removed. This shouldn't happen")
        }))
    }
}
