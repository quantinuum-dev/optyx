use std::collections::HashSet;

use quizx::scalar::*;


/// Type representing vertices in a graph
pub type V = usize;

/// A Port specifies a connection point on a vertex
pub trait Port: Eq + std::hash::Hash + Clone + Copy + std::fmt::Debug {
    /// Creates a new port refereing to an index on a vertex
    fn new(v: V, idx: usize) -> Self;

    /// Returns the vertex this port belongs to
    fn get_vert(&self) -> V;

    /// Returns the index indentifying this port on the vertex
    fn get_idx(&self) -> usize;
}


/// A Port identifying a vertex input
#[derive(Debug,Clone,Copy,PartialEq,Eq,Hash)]
pub struct InPort {
    vert: V,
    idx: usize
}

impl Port for InPort {
    fn new(v: V, idx: usize) -> Self {
        InPort {vert: v, idx: idx}
    }

    fn get_vert(&self) -> V {
        self.vert
    }

    fn get_idx(&self) -> usize {
        self.idx
    }
}


/// A Port identifying a vertex output
#[derive(Debug,Clone,Copy,PartialEq,Eq,Hash)]
pub struct OutPort {
    vert: V,
    idx: usize
}

impl Port for OutPort {
    fn new(v: V, idx: usize) -> Self {
        OutPort {vert: v, idx: idx}
    }

    fn get_vert(&self) -> V {
        self.vert
    }

    fn get_idx(&self) -> usize {
        self.idx
    }
}



/// Trait to generalise over the different types of vertices in graphs
pub trait VType: Clone + Copy + Send + Sync + std::fmt::Debug {
    /// Returns the number of expected inputs for a vertex of this type or `None` if vertices of
    /// this type do not have a fixed number of inputs.
    fn num_expected_inputs(&self) -> Option<usize>;

    /// Returns the number of expected outputs for a vertex of this type or `None` if vertices of
    /// this type do not have a fixed number of outputs.
    fn num_expected_outputs(&self) -> Option<usize>;
}

/// Trait to generalise over the different types of edges in graphs. At the moment this is just a
/// tag trait.
pub trait EType: Clone + Copy + Send + Sync + std::fmt::Debug {}


/// Abstract graph parametrised over vertex and edge types
pub trait Graph<VTy: VType, ETy: EType>: Clone + Sized + Send + Sync + std::fmt::Debug {
    /// Create an empty graph
    fn new() -> Self;

    fn vertices(&self) -> VIter<VTy>;

    fn edges(&self) -> EIter<ETy>;

    fn add_vertex(&mut self, vty: VTy) -> (V, Vec<InPort>, Vec<OutPort>);

    fn remove_vertex(&mut self, v: V);

    fn in_ports(&self, v: V) -> Vec<InPort>;

    fn out_ports(&self, v: V) -> Vec<OutPort>;

    fn add_in_port(&mut self, v: V) -> InPort;

    fn add_out_port(&mut self, v: V) -> OutPort;

    fn add_edge(&mut self, u: OutPort, v: InPort, ety: ETy);

    fn remove_edge(&mut self, u: OutPort, v: InPort);

    fn edge_type_opt(&self, u: OutPort, v: InPort) -> Option<ETy>;

    fn set_edge_type(&mut self, u: OutPort, v: InPort, ety: ETy);

    fn vertex_type(&self, v: V) -> VTy;

    fn set_vertex_type(&mut self, v: V, vty: VTy);

    fn incident_edges_in(&self, v: V) -> Vec<(OutPort, InPort, ETy)>;

    fn incident_edges_out(&self, v: V) -> Vec<(OutPort, InPort, ETy)>;

    fn scalar(&self) -> &ScalarN;

    fn scalar_mut(&mut self) -> &mut ScalarN;

    fn edge_type(&self, s: OutPort, t: InPort) -> ETy {
        self.edge_type_opt(s,t).expect("Edge not found")
    }

    fn to_dot(&self) -> String {
        let mut dot = String::from("digraph {\n");
        for v in self.vertices() {
            let t = self.vertex_type(v);
            dot += &format!("  {} [label=\"{}: {:?}\"", v, v, t);
            dot += "]\n";
        }

        dot += "\n";

        for (s, t, ety) in self.edges() {
            dot += &format!("  {} -> {}", s.vert, t.vert);
            dot += &format!(" [label=\"{:?}\"]", ety);
            dot += "\n";
        }

        dot += "}\n";

        dot
    }

    fn topsort(&self) -> Vec<V> {
        // Bool flag indicates whether all children have been visited once the vertex is on top of
        // the stack
        let mut stack: Vec<(bool, V)> = Vec::new(); 
        let mut visited: HashSet<V> = HashSet::new();
        let mut sort: Vec<V> = Vec::new();

        for v in self.vertices() {
            if !visited.contains(&v) {
                stack.push((false, v));
            }

            while !stack.is_empty() {
                let (all_children_done, u) = stack.pop().unwrap();
                if all_children_done {
                    sort.push(u);
                    continue;
                }
                if visited.contains(&u) {
                    continue;
                }
                visited.insert(u);
                stack.push((true, u));
                for (_, child, _) in self.incident_edges_out(u) {
                    if visited.contains(&child.vert) {
                        stack.push((false, child.vert));
                    }
                }
            }
        }

        sort
    }

}


pub type Table<T> = Vec<Option<T>>;

#[derive(Debug,Clone)]
pub struct VecGraph<VTy: VType, ETy: EType> {
    vtable: Table<VTy>,
    in_edges: Table<Vec<Option<(OutPort, ETy)>>>,
    out_edges: Table<Vec<Option<(InPort, ETy)>>>,
    holes: Vec<V>,
    num_verts: usize,
    num_edges: usize,
    scalar: ScalarN
}


impl<VTy: VType, ETy: EType> Graph<VTy, ETy> for VecGraph<VTy, ETy> {
    fn new() -> Self {
        VecGraph {
            vtable: Vec::new(),
            in_edges: Vec::new(),
            out_edges: Vec::new(),
            holes: Vec::new(),
            num_verts: 0,
            num_edges: 0,
            scalar: Scalar::one()
        }
    }

    fn vertices(&self) -> VIter<VTy> {
        VIter::Vec(self.num_verts, self.vtable.iter().enumerate())
    }

    fn edges(&self) -> EIter<ETy> {
        EIter::Vec(self.num_edges, self.out_edges.iter().enumerate(), None)
    }

    fn add_vertex(&mut self, vty: VTy) -> (V, Vec<InPort>, Vec<OutPort>) {
        self.num_verts += 1;
        let vert: V;
        if let Some(v) = self.holes.pop() {
            self.vtable[v] = Some(vty);
            vert = v;
        } else {
            self.vtable.push(Some(vty));
            self.in_edges.push(Some(Vec::new()));
            self.out_edges.push(Some(Vec::new()));
            vert = self.vtable.len() - 1;
        }

        let num_inputs = vty.num_expected_inputs().unwrap_or(0);
        let num_outputs = vty.num_expected_outputs().unwrap_or(0);

        self.in_edges[vert] = Some(vec![None; num_inputs]);
        self.out_edges[vert] = Some(vec![None; num_outputs]);
        
        let in_ports: Vec<InPort> = num::range(0, num_inputs)
            .map(|i| InPort {vert: vert, idx: i}).collect();

        let out_ports: Vec<OutPort> = num::range(0, num_outputs)
            .map(|i| OutPort {vert: vert, idx: i}).collect();

        (vert, in_ports, out_ports)
    }

    fn in_ports(&self, v: V) -> Vec<InPort> {
        if let Some(Some(inputs)) = self.in_edges.get(v) {
            Vec::from_iter(inputs.iter().enumerate().filter_map(|(idx, o)| o.map(|_| InPort::new(v, idx))))
        } else {
            panic!("Vertex not found");
        }
    }

    fn out_ports(&self, v: V) -> Vec<OutPort> {
        if let Some(Some(outputs)) = self.out_edges.get(v) {
            Vec::from_iter(outputs.iter().enumerate().filter_map(|(idx, o)| o.map(|_| OutPort::new(v, idx))))
        } else {
            panic!("Vertex not found");
        }
    }

    fn remove_vertex(&mut self, v: V) {
        for (other, this, _) in self.incident_edges_in(v) {
            self.remove_edge(other, this);
        }
        for (this, other, _) in self.incident_edges_out(v) {
            self.remove_edge(this, other);
        }

        self.num_verts -= 1;
        self.holes.push(v);
        self.vtable[v] = None;
        self.in_edges[v] = None;
        self.out_edges[v] = None;
    }

    fn add_in_port(&mut self, v: V) -> InPort {
        if let Some(Some(inputs)) = self.in_edges.get_mut(v) {
            inputs.push(None);
            InPort::new(v, inputs.len() - 1)
        } else {
            panic!("Vertex not found");
        }
    }

    fn add_out_port(&mut self, v: V) -> OutPort {
        if let Some(Some(outputs)) = self.out_edges.get_mut(v) {
            outputs.push(None);
            OutPort::new(v, outputs.len() - 1)
        } else {
            panic!("Vertex not found");
        }
    }

    fn add_edge(&mut self, u: OutPort, v: InPort, ety: ETy) {
        if let Some(Some(outputs)) = self.out_edges.get_mut(u.vert) {
            outputs[u.idx] = Some((v, ety));
        } else {
            panic!("Source vertex not found");
        }

        if let Some(Some(inputs)) = self.in_edges.get_mut(v.vert) {
            inputs[v.idx] = Some((u, ety));
        } else {
            panic!("Target vertex not found");
        }

        self.num_edges += 1;
    }

    fn remove_edge(&mut self, u: OutPort, v: InPort) {
        if let Some(Some(outputs)) = self.out_edges.get_mut(u.vert) {
            outputs[u.idx] = None;
        } else {
            panic!("Source vertex not found");
        }

        if let Some(Some(inputs)) = self.in_edges.get_mut(v.vert) {
            inputs[v.idx] = None;
        } else {
            panic!("Target vertex not found");
        }

        self.num_edges -= 1;
    }

    fn edge_type_opt(&self, u: OutPort, _: InPort) -> Option<ETy> {
        if let Some(Some(outputs)) = self.out_edges.get(u.vert) {
            outputs[u.idx].map(|x| x.1)
        } else {
            None
        }
    }

    fn set_edge_type(&mut self, u: OutPort, v: InPort, ety: ETy) {
        if let Some(Some(outputs)) = self.out_edges.get_mut(u.vert) {
            outputs[u.idx] = Some((v, ety));
            
        } else {
            panic!("Source vertex not found");
        }

        if let Some(Some(inputs)) = self.in_edges.get_mut(v.vert) {
            inputs[v.idx] = Some((u, ety));
        } else {
            panic!("Target vertex not found");
        }
    }

    fn vertex_type(&self, v: V) -> VTy {
        self.vtable.get(v).expect("Vertex not found").expect("Vertex not found")
    }

    fn set_vertex_type(&mut self, v: V, vty: VTy) {
        *self.vtable.get_mut(v).expect("Vertex not found") = Some(vty);
    }

    fn incident_edges_in(&self, v: V) -> Vec<(OutPort, InPort, ETy)> {
        if let Some(Some(inputs)) = self.in_edges.get(v) {
            Vec::from_iter(inputs.iter().enumerate()
                .filter_map(|(idx, o)| 
                    o.map(|(out, ety)| (out, InPort {vert: v, idx: idx}, ety))))
        } else {
            panic!("Vertex not found")
        }
    }

    fn incident_edges_out(&self, v: V) -> Vec<(OutPort, InPort, ETy)> {
        if let Some(Some(outputs)) = self.out_edges.get(v) {
            Vec::from_iter(outputs.iter().enumerate()
                .filter_map(|(idx, o)| 
                    o.map(|(inp, ety)| (OutPort {vert: v, idx: idx},inp, ety))))
        } else {
            panic!("Vertex not found")
        }
    }

    fn scalar(&self) -> &ScalarN {
        &self.scalar
    }

    fn scalar_mut(&mut self) -> &mut ScalarN {
        &mut self.scalar
    }

}

impl<VTy: VType, ETy: EType> VecGraph<VTy, ETy> {
    pub fn map_verts<NewVTy: VType>(self, f: impl Fn(VTy) -> NewVTy) -> VecGraph<NewVTy, ETy> {
        VecGraph {
            vtable: self.vtable.iter().map(|&o| o.map(|v| f(v))).collect(),
            in_edges: self.in_edges,
            out_edges: self.out_edges,
            holes: self.holes,
            num_verts: self.num_verts,
            num_edges: self.num_edges,
            scalar: self.scalar
        }
    }

    pub fn map_edges<NewETy: EType>(self, f: Box<dyn Fn(ETy) -> NewETy>) -> VecGraph<VTy, NewETy> {
        VecGraph {
            vtable: self.vtable,
            in_edges: self.in_edges.iter().map(|o| o.clone().map(|inputs| inputs.iter().map(|o| o.map(|(v, e)| (v, f(e)))).collect())).collect(),
            out_edges: self.out_edges.iter().map(|o| o.clone().map(|outputs| outputs.iter().map(|o| o.map(|(v, e)| (v, f(e)))).collect())).collect(),
            holes: self.holes,
            num_verts: self.num_verts,
            num_edges: self.num_edges,
            scalar: self.scalar
        }
    }
}

// pub trait GraphMap<VertFun, NewVert, EdgeFun, NewEdge> {
//     fn map_verts(self, f: VertFun) -> NewVert;
//     fn map_edges(self, f: EdgeFun) -> NewEdge;
// }

// impl<OldVTy: VType, NewVTy: VType, OldETy: EType, NewETy: EType> GraphMap<Box<dyn Fn(OldVTy) -> NewVTy>, VecGraph<NewVTy, OldETy>, Box<dyn Fn(OldETy) -> NewETy>, VecGraph<OldVTy, NewETy>> for VecGraph<OldVTy, OldETy> {
//     fn map_verts(self, f: Box<dyn Fn(OldVTy) -> NewVTy>) -> VecGraph<NewVTy, OldETy> {
//         VecGraph {
//             vtable: self.vtable.iter().map(|&o| o.map(f)).collect(),
//             vinputs: self.vinputs,
//             voutputs: self.voutputs,
//             holes: self.holes,
//             num_verts: self.num_verts,
//             num_edges: self.num_edges
//         }
//     }

//     fn map_edges(self, f: Box<dyn Fn(OldETy) -> NewETy>) -> VecGraph<OldVTy, NewETy> {
//         VecGraph {
//             vtable: self.vtable,
//             vinputs: self.vinputs.iter().map(|&o| o.map(|inputs| inputs.iter().map(|(v, e)| (*v, f(*e))).collect())).collect(),
//             voutputs: self.voutputs.iter().map(|&o| o.map(|inputs| inputs.iter().map(|(v, e)| (*v, f(*e))).collect())).collect(),
//             holes: self.holes,
//             num_verts: self.num_verts,
//             num_edges: self.num_edges
//         }
//     }
// }


/// Collection of vertex enumerators for the different graph implementations. At the moment this is
/// only the `VecGraph`.
pub enum VIter<'a, VTy: VType> {
    /// Vertex iterator for `VecGrap`
    Vec(usize,std::iter::Enumerate<std::slice::Iter<'a,Option<VTy>>>),
}

/// Collection of edge enumerators for the different graph implementations. At the moment this is
/// only the `VecGraph`.
pub enum EIter<'a, ETy: EType> {
    /// Edge iterator for `VecGrap`
    Vec(usize,  // Number of edges
        std::iter::Enumerate<std::slice::Iter<'a,Option<Vec<Option<(InPort, ETy)>>>>>,  // Outer iterator
        Option<(V,  // Vertex whose connections we're currently visiting
                usize,  // Current port index
                std::slice::Iter<'a,Option<(InPort, ETy)>>)>)  // Inner iterator
}

impl<'a, VTy: VType> Iterator for VIter<'a, VTy> {
    type Item = V;
    fn next(&mut self) -> Option<V> {
        match self {
            VIter::Vec(_,inner)  => {
                let mut next = inner.next();

                // skip over "holes", i.e. vertices that have been deleted
                while next.is_some() && !next.unwrap().1.is_some() {
                    next = inner.next();
                }

                match next {
                    Some((v, Some(_))) => Some(v),
                    Some((_, None)) => panic!("encountered deleted vertex in VIter"), // should never happen
                    None => None,
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = match self {
            VIter::Vec(sz,_)  => *sz
        };
        (len, Some(len))
    }
}

impl<'a, ETy: EType> Iterator for EIter<'a, ETy> {
    type Item = (OutPort,InPort,ETy);
    fn next(&mut self) -> Option<(OutPort,InPort,ETy)> {
        match self {
            EIter::Vec(_,outer,inner) => {
                loop {
                    // "inner" iterates the neighborhood of a single vertex
                    if let Some((v, idx, iter)) = inner {
                        let mut inner_next = iter.next();
                        *idx += 1;

                        // skip over ports that are not used
                        while inner_next.is_some() && inner_next.unwrap().is_none() {
                            inner_next = iter.next();
                            *idx += 1;
                        }

                        if let Some(Some((u, et))) = inner_next {
                            return Some((OutPort {vert: *v, idx: *idx-1}, *u, *et));
                        }
                    }

                    // if we get to here, either we are a brand new iterator or we've run out of
                    // edges next to the current vertex, so we need to proceed to the next one
                    let mut outer_next = outer.next();

                    // skip over "holes", i.e. vertices that have been deleted
                    while outer_next.is_some() && outer_next.unwrap().1.is_none() {
                        outer_next = outer.next();
                    }

                    match outer_next {
                        // proceed to the next vertex and loop
                        Some((v, Some(tab))) => {
                            *inner = Some((v, 0, tab.iter()));
                        },
                        // should never happen
                        Some((_, None)) => panic!("encountered deleted vertex in EIter"),
                        // out of vertices, so terminate iteration
                        None => { return None; }
                    }
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = match self {
            EIter::Vec(sz, ..)  => *sz
        };
        (len, Some(len))
    }
}
