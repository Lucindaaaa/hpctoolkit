// -*-Mode: C++;-*-

// * BeginRiceCopyright *****************************************************
//
// $HeadURL$
// $Id$
//
// --------------------------------------------------------------------------
// Part of HPCToolkit (hpctoolkit.org)
//
// Information about sources of support for research and development of
// HPCToolkit is at 'hpctoolkit.org' and in 'README.Acknowledgments'.
// --------------------------------------------------------------------------
//
// Copyright ((c)) 2002-2017, Rice University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of Rice University (RICE) nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// This software is provided by RICE and contributors "as is" and any
// express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular
// purpose are disclaimed. In no event shall RICE or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services; loss of use, data, or profits; or
// business interruption) however caused and on any theory of liability,
// whether in contract, strict liability, or tort (including negligence
// or otherwise) arising in any way out of the use of this software, even
// if advised of the possibility of such damage.
//
// ******************************************************* EndRiceCopyright *

/* 
 * File:   DifferenceQuantifier.cpp
 * Author: Lai Wei <lai.wei@rice.edu>
 *
 * Created on April 23, 2018, 12:47 AM
 */

#include <algorithm>
using std::max;
using std::min;

#include "TraceAnalysisCommon.hpp"
#include "DifferenceQuantifier.hpp"
#include "data/TCT-Node.hpp"

namespace TraceAnalysis {
  // Compute difference between two duration ranges (min1, max1) and (min2, max2).
  // Noise from sampling is considered.
  // Noise should be no more than MAX_SAMPLE_NOISE times sampling period.
  Time computeRangeDiff(Time min1, Time max1, Time min2, Time max2, Time samplingPeriod) {
    // Adjust duration ranges when noise is greater than MAX_SAMPLE_NOISE times sampling period.
    if (max1 - min1 > MAX_SAMPLE_NOISE * samplingPeriod) {
      Time mid1 = (max1 + min1) / 2;
      min1 = mid1 - MAX_SAMPLE_NOISE * samplingPeriod / 2;
      max1 = mid1 + MAX_SAMPLE_NOISE * samplingPeriod / 2;
    }
    if (max2 - min2 > MAX_SAMPLE_NOISE * samplingPeriod) {
      Time mid2 = (max2 + min2) / 2;
      min2 = mid2 - MAX_SAMPLE_NOISE * samplingPeriod / 2;
      max2 = mid2 + MAX_SAMPLE_NOISE * samplingPeriod / 2;
    }
    
    // If the two ranges don't intersect, return their minimum possible difference.
    if (max2 < min1) return min1 - max2;
    if (max1 < min2) return min2 - max1;
    
    // If the two ranges intersect, return 0.
    return 0;
  }
  
  TCTANode* DifferenceQuantifier::mergeNode(const TCTANode* node1, long weight1, const TCTANode* node2, long weight2, 
          bool ifAccumulate, bool isScoreOnly, bool isSum) {
    if (!(node1->id == node2->id))
      print_msg(MSG_PRIO_MAX, "ERROR: merging two nodes with different id: %s vs %s.\n", 
              node1->id.toString().c_str(), node2->id.toString().c_str());
    
    if (node1->getDepth() != node2->getDepth())
      print_msg(MSG_PRIO_HIGH, "ERROR: merging node at difference depth %d vs %d.\n", node1->getDepth(), node2->getDepth());
    
    if (node1->type == TCTANode::NonCFGProf || node2->type == TCTANode::NonCFGProf) {
      TCTNonCFGProfileNode* prof1 = node1->type == TCTANode::NonCFGProf ? (TCTNonCFGProfileNode*)node1 : TCTNonCFGProfileNode::newNonCFGProfileNode(node1);
      TCTNonCFGProfileNode* prof2 = node2->type == TCTANode::NonCFGProf ? (TCTNonCFGProfileNode*)node2 : TCTNonCFGProfileNode::newNonCFGProfileNode(node2);
      TCTNonCFGProfileNode* merged = mergeProfileNode(prof1, weight1, prof2, weight2, ifAccumulate, isScoreOnly, isSum);
      if (node1->type != TCTANode::NonCFGProf) delete prof1;
      if (node2->type != TCTANode::NonCFGProf) delete prof2;
      return merged;
    }
    else {
      if (node1->type != node2->type) {
        // if one of them is a loop, use its profile node instead.
        if (node1->type == TCTANode::Loop)
          return mergeNode(((TCTLoopNode*)node1)->getProfileNode(), weight1, node2, weight2, ifAccumulate, isScoreOnly, isSum);
        if (node2->type == TCTANode::Loop)
          return mergeNode(node1, weight1, ((TCTLoopNode*)node2)->getProfileNode(), weight2, ifAccumulate, isScoreOnly, isSum);
        
        print_msg(MSG_PRIO_MAX, "ERROR: merging two nodes but their types don't match %d vs %d.\n", node1->type, node2->type);
        return node1->voidDuplicate();
      }
      if (node1->type == TCTANode::Loop) {
        return mergeLoopNode((TCTLoopNode*)node1, weight1, (TCTLoopNode*)node2, weight2, ifAccumulate, isScoreOnly, isSum);
      }
      else 
        return mergeACFGNode((TCTACFGNode*)node1, weight1, (TCTACFGNode*)node2, weight2, ifAccumulate, isScoreOnly, isSum);
    }
  }

  TCTNonCFGProfileNode* DifferenceQuantifier::mergeProfileNode(const TCTNonCFGProfileNode* prof1, long weight1, const TCTNonCFGProfileNode* prof2, long weight2, 
          bool ifAccumulate, bool isScoreOnly, bool isSum) {
    TCTNonCFGProfileNode* mergedProf = (TCTNonCFGProfileNode*)prof1->voidDuplicate();
    if (!isSum) {
      mergedProf->getTime().setAsAverageTime(prof1->getTime(), weight1, prof2->getTime(), weight2);
      if (!isScoreOnly) {
        mergedProf->setWeight(weight1 + weight2);
        mergedProf->setRetCount(prof1->getRetCount() + prof2->getRetCount());
        mergedProf->setDerivedSemanticLabel(prof1->getDerivedSemanticLabel() & prof2->getDerivedSemanticLabel());
        mergedProf->getPerfLossMetric().setDuratonMetric(prof1->getPerfLossMetric(), prof2->getPerfLossMetric());
      }
    }
    else {
      mergedProf->getTime().clear();
      mergedProf->getTime().addTime(prof1->getTime());
      mergedProf->getTime().addTime(prof2->getTime());
      mergedProf->setWeight(weight1);
      mergedProf->setRetCount(prof1->getRetCount() + prof2->getRetCount());
    }
    
    /**
     * When accumulating difference scores, the inclusive difference score of a node is the sum of difference scores in its subtree.
     * Inclusive difference score of the merged profile node consists of the following components --
     *  1) the inclusive difference score of all children, which can be divided to:
     *    1.1) the inclusive difference score of all children in prof1;
     *    1.2) the inclusive difference score of all children in prof2;
     *    1.3) the difference score of children between prof1 and prof2;
     *  2) the difference of exclusive duration among all nodes represented by the merged profile node.
     *    2.1) the difference of exclusive duration among all nodes represented by prof1
     *    2.2) the difference of exclusive duration among all nodes represented by prof2
     *    2.3) the difference of exclusive duration between nodes represented by prof1 and prof2
     *  
     *  When not accumulating, only 1.3) and 2.3) needs to be added up.
     */
    Time samplingPeriod = mergedProf->getSamplingPeriod();
    double inclusiveDiff = 0;
    
    const map<TCTID, TCTNonCFGProfileNode*>& map1 = prof1->getChildMap();
    const map<TCTID, TCTNonCFGProfileNode*>& map2 = prof2->getChildMap();
    
    for (auto it = map1.begin(); it != map1.end(); it++) {
      TCTNonCFGProfileNode* child1 = it->second;
      // Children that both profile nodes have
      if (map2.find(child1->id) != map2.end()) {
        TCTNonCFGProfileNode* child2 = map2.find(child1->id)->second;
        TCTNonCFGProfileNode* mergedChild = (TCTNonCFGProfileNode*)mergeNode(child1, weight1, child2, weight2, ifAccumulate, isScoreOnly, isSum);
        inclusiveDiff += mergedChild->getDiffScore().getInclusive(); // belongs to 1.3) when not accumulating and 1) when accumulating
        
        if (isScoreOnly) delete mergedChild;
        else mergedProf->addChild(mergedChild);
      }
      else { // Children that only profile 1 has.
        TCTNonCFGProfileNode* child2 = (TCTNonCFGProfileNode*) child1->voidDuplicate();
        child2->setWeight(weight2);
        if (!isSum) child2->getTime().setDuration(-samplingPeriod, samplingPeriod);
        child2->initPerfLossMetric();
        
        TCTNonCFGProfileNode* mergedChild = (TCTNonCFGProfileNode*)mergeNode(child1, weight1, child2, weight2, ifAccumulate, isScoreOnly, isSum);
        inclusiveDiff += mergedChild->getDiffScore().getInclusive(); // belongs to 1.3) when not accumulating and 1) when accumulating
        
        if (isScoreOnly) delete mergedChild;
        else mergedProf->addChild(mergedChild);
        
        delete child2;
      }
    }
    
    for (auto it = map2.begin(); it != map2.end(); it++)
      // Children that only profile 2 has.
      if (map1.find(it->first) == map1.end()) {
        TCTNonCFGProfileNode* child2 = it->second;
        TCTNonCFGProfileNode* child1 = (TCTNonCFGProfileNode*) child2->voidDuplicate();
        child1->setWeight(weight1);
        if (!isSum) child1->getTime().setDuration(-samplingPeriod, samplingPeriod);
        child1->initPerfLossMetric();
        
        TCTNonCFGProfileNode* mergedChild = (TCTNonCFGProfileNode*)mergeNode(child1, weight1, child2, weight2, ifAccumulate, isScoreOnly, isSum);
        inclusiveDiff += mergedChild->getDiffScore().getInclusive(); // belongs to 1.3) when not accumulating and 1) when accumulating

        if (isScoreOnly) delete mergedChild;
        else mergedProf->addChild(mergedChild);
        
        delete child1;
      }
    
    if (!isSum) {
      Time minExclusive1, maxExclusive1, minExclusive2, maxExclusive2;
      prof1->getExclusiveDuration(minExclusive1, maxExclusive1);
      prof2->getExclusiveDuration(minExclusive2, maxExclusive2);

      inclusiveDiff += (double)computeRangeDiff(minExclusive1, maxExclusive1, minExclusive2, maxExclusive2, samplingPeriod)
              * (double) weight1 * (double) weight2;  // belongs to 2.3)

      /**
       * The exclusive difference score of a node is the sum of difference at this node (not counting is children),
       * which is the sum of difference of inclusive duration among all nodes represented by the merged profile node.
       * When accumulating, exclusive difference score of the merged profile node consists of the following components -- 
       *  1) the difference of inclusive duration among all nodes represented by prof1
       *  2) the difference of inclusive duration among all nodes represented by prof2
       *  3) the difference of inclusive duration between nodes represented by prof1 and prof2
       * 
       * When not accumulating, only 3) is needed.
       */
      double exclusiveDiff = (double)computeRangeDiff(prof1->getMinDuration(), prof1->getMaxDuration(), 
                                                      prof2->getMinDuration(), prof2->getMaxDuration(), samplingPeriod)
              * (double) weight1 * (double) weight2; // Add 3)

      if (ifAccumulate) {
        // Add 1) and 2) to exclusive diff score.
        exclusiveDiff += prof1->getDiffScore().getExclusive() + prof2->getDiffScore().getExclusive();

        // The above code has added 1) and 2.3) to inclusive diff score. Add 2.1) and 2.2) to inclusive diff score below.
        // Add 2.1)
        inclusiveDiff += prof1->getDiffScore().getInclusive();
        for (auto it = map1.begin(); it != map1.end(); it++)
          inclusiveDiff -= it->second->getDiffScore().getInclusive();

        // Add 2.2)
        inclusiveDiff += prof2->getDiffScore().getInclusive();
        for (auto it = map2.begin(); it != map2.end(); it++)
          inclusiveDiff -= it->second->getDiffScore().getInclusive();
      }

      /* Inclusive diff score should be no less than "exclusive diff score" plus minimum accumulation.
       * (As we hide noises from sampling, the resulting inclusive diff score could be smaller than 
       * the minimum threshold)
       */
      double minInclusiveDiff = (double)computeRangeDiff(prof1->getMinDuration(), prof1->getMaxDuration(), 
                                                      prof2->getMinDuration(), prof2->getMaxDuration(), samplingPeriod)
              * (double) weight1 * (double) weight2;
      if (ifAccumulate)
        minInclusiveDiff += prof1->getDiffScore().getInclusive() + prof2->getDiffScore().getInclusive();

      if (inclusiveDiff < minInclusiveDiff) inclusiveDiff = minInclusiveDiff;

      mergedProf->getDiffScore().setScores(inclusiveDiff, exclusiveDiff);
    }
    
    return mergedProf;
  }
  
  class EdgeManager {
  private:
    // Stores edges whose destination node haven't been finalized.
    //          ID of target        ID of source   
    unordered_map<TCTID, unordered_map<TCTID, TCTACFGNode::Edge*>> pendingEdges;
    
    // Stores edges that are completed.
    vector<TCTACFGNode::Edge*> completedEdges;
    
  public:
    EdgeManager() {}
    
    // Insert a new pending edge and set its source to srcNode.
    void insertPendingEdge(TCTACFGNode::Edge* newEdge, TCTANode* srcNode) {
      newEdge->setSrc(srcNode); // set the source of the new edge.
      
      if (pendingEdges.find(newEdge->getDst()->id) == pendingEdges.end())
        pendingEdges[newEdge->getDst()->id] = unordered_map<TCTID, TCTACFGNode::Edge*>();
      
      unordered_map<TCTID, TCTACFGNode::Edge*>& srcMap = pendingEdges[newEdge->getDst()->id];
      if (srcMap.find(newEdge->getSrc()->id) == srcMap.end())
        srcMap[newEdge->getSrc()->id] = newEdge;
      else {
        srcMap[newEdge->getSrc()->id]->addWeight(newEdge->getWeight());
        delete newEdge;
      }
    }
    
    // For every pending edge [src -> dst], check and see if there is a control flow from src -> mid and mid -> dst.
    // If so, split the edge [src -> dst] into two edges -- [src -> mid] and [mid -> dst].
    void checkPendingEdges(CFGAGraph* cfg, TCTANode* midNode) {
      vector<TCTACFGNode::Edge*> splitEdges; // Temporarily store split edges.
      
      for (auto mit = pendingEdges.begin(); mit != pendingEdges.end(); mit++) {
        const TCTANode* dst = mit->second.begin()->second->getDst();
        if (dst->id == TCTACFGNode::dummyEndNode.id || cfg->compareChild(midNode->ra, dst->ra) == -1) { // if there is a control flow from mid -> dst.
          for (auto it = mit->second.begin(); it != mit->second.end();) {
            const TCTANode* src = it->second->getSrc(); 
            if (src->id == TCTACFGNode::dummyBeginNode.id || cfg->compareChild(midNode->ra, src->ra) == 1) { // if there is a control flow from src -> mid
              // Split the edge into two
              // The first edge is src->mid
              TCTACFGNode::Edge* edge1 = it->second->duplicate();
              edge1->setDst(midNode);
              splitEdges.push_back(edge1);

              // The second edge is mid->dst
              TCTACFGNode::Edge* edge2 = it->second;
              edge2->setSrc(midNode);
              splitEdges.push_back(edge2);

              it = mit->second.erase(it);
            }
            else
              it++;
          }
        }
      }
      
      while (!splitEdges.empty()) {
        insertPendingEdge(splitEdges.back(), splitEdges.back()->getSrc());
        splitEdges.pop_back();
      }
    }
    
    // Complete all pending edges whose destination node matches dstNode.
    void completePendingEdge(TCTANode* dstNode) {
      if (pendingEdges.find(dstNode->id) == pendingEdges.end()) return;

      for (auto it = pendingEdges[dstNode->id].begin(); it != pendingEdges[dstNode->id].end(); it++) {
        TCTACFGNode::Edge* edge = it->second;
        edge->setDst(dstNode); // set the destination of the edge.
        completedEdges.push_back(edge);
      }
      
      pendingEdges.erase(dstNode->id);
    }
    
    vector<TCTACFGNode::Edge*>& getAllEdges() {
      return completedEdges;
    }
    
    // This can only be called when edges are not inserted to any CFG node.
    void deallocateAllEdges() {
      for (auto edge : completedEdges)
        delete edge;
      
      for (auto mit = pendingEdges.begin(); mit != pendingEdges.end(); mit++)
        for (auto it = mit->second.begin(); it != mit->second.end(); it++)
          delete it->second;
    }
  };
  
  TCTANode* DifferenceQuantifier::mergeACFGNode(const TCTACFGNode* node1, long weight1, const TCTACFGNode* node2, long weight2, 
          bool ifAccumulate, bool isScoreOnly, bool isSum) {
    TCTACFGNode* mergedCFG = (TCTACFGNode*) node1->voidDuplicate();
    if (!isSum) {
      mergedCFG->getTime().setAsAverageTime(node1->getTime(), weight1, node2->getTime(), weight2);
      if (!isScoreOnly) {
        mergedCFG->setWeight(weight1 + weight2);
        mergedCFG->setRetCount(node1->getRetCount() + node2->getRetCount());
        mergedCFG->setDerivedSemanticLabel(node1->getDerivedSemanticLabel() & node2->getDerivedSemanticLabel());
        mergedCFG->getPerfLossMetric().setDuratonMetric(node1->getPerfLossMetric(), node2->getPerfLossMetric());
      }
    }
    else {
      mergedCFG->toCFGProfile();
      mergedCFG->getTime().clear();
      mergedCFG->getTime().addTime(node1->getTime());
      mergedCFG->getTime().addTime(node2->getTime());
      mergedCFG->setWeight(weight1);
      mergedCFG->setRetCount(node1->getRetCount() + node2->getRetCount());
    }
    
    /**
     * Inclusive difference score of the merged CFG consists of the following components --
     *  1) the inclusive difference score of all children, which can be divided to:
     *    1.1) the inclusive difference score of all children in node1;
     *    1.2) the inclusive difference score of all children in node2;
     *    1.3) the difference score of children between node1 and node2;
     * 
     *  2) the difference of exclusive duration among all nodes represented by the merged CFG node.
     *    2.1) the difference of exclusive duration among all nodes represented by node1
     *    2.2) the difference of exclusive duration among all nodes represented by node2
     *    2.3) the difference of exclusive duration between nodes represented by node1 and node2
     *  
     *  When not accumulating, only 1.3) and 2.3) needs to be added up.
     */
    double inclusiveDiff = 0;
    Time samplingPeriod = mergedCFG->getSamplingPeriod();
    int k1 = 0, k2 = 0;
    
    EdgeManager em;
    if (!isScoreOnly) {
      // Process entry edges for node1 and node2
      if (node1->getNumChild() > 0)
        for (auto it = node1->getEntryEdges().begin(); it != node1->getEntryEdges().end(); it++)
          em.insertPendingEdge((*it)->duplicate(), &TCTACFGNode::dummyBeginNode);
      else {
        TCTACFGNode::Edge* edge = new TCTACFGNode::Edge(&TCTACFGNode::dummyBeginNode, &TCTACFGNode::dummyEndNode, weight1);
        em.insertPendingEdge(edge, &TCTACFGNode::dummyBeginNode);
      }
      
      if (node2->getNumChild() > 0)
        for (auto it = node2->getEntryEdges().begin(); it != node2->getEntryEdges().end(); it++)
          em.insertPendingEdge((*it)->duplicate(), &TCTACFGNode::dummyBeginNode);
      else {
        TCTACFGNode::Edge* edge = new TCTACFGNode::Edge(&TCTACFGNode::dummyBeginNode, &TCTACFGNode::dummyEndNode, weight2);
        em.insertPendingEdge(edge, &TCTACFGNode::dummyBeginNode);
      }
    }
    
    while (k1 < node1->getNumChild() || k2 < node2->getNumChild()) {
      // When children of trace1 and trace2 match
      if (k1 < node1->getNumChild() && k2 < node2->getNumChild()
              && node1->getChild(k1)->id == node2->getChild(k2)->id) {
        const TCTANode* child1 = node1->getChild(k1);
        const TCTANode* child2 = node2->getChild(k2);
        
        // Add child diff: 1.3) when not accumulating and 1) when accumulating
        TCTANode* mergedChild = mergeNode(child1, weight1, child2, weight2, ifAccumulate, isScoreOnly, isSum);
        inclusiveDiff += mergedChild->getDiffScore().getInclusive();
        
        if (isScoreOnly) delete mergedChild;
        else {
          mergedCFG->addChild(mergedChild);
          // Process inEdges of the merged child
          em.completePendingEdge(mergedChild);
          // Process outEdges of child1 and child2
          for (auto it = node1->getOutEdges(k1).begin(); it != node1->getOutEdges(k1).end(); it++)
            em.insertPendingEdge((*it)->duplicate(), mergedChild);
          for (auto it = node2->getOutEdges(k2).begin(); it != node2->getOutEdges(k2).end(); it++)
            em.insertPendingEdge((*it)->duplicate(), mergedChild);
        }
        k1++; k2++;
      }
      // When children of trace1 and trace2 don't match
      else {
        int compare = 0; // compare < 0 means trace1[k1] should be processed first; > 0 otherwise.
        if (k1 == node1->getNumChild()) compare = 1;
        if (k2 == node2->getNumChild()) compare = -1;
        
        if (node1->cfgGraph == NULL // When cfgGraph is NULL 
            || (compare == 0  
               && ( !node1->cfgGraph->hasChild(node1->getChild(k1)->ra) // RA of node1[k1] is not found in cfgGraph
                    || !node1->cfgGraph->hasChild(node2->getChild(k2)->ra) // or RA of node2[k2] is not found in cfgGraph
               ))) { // Return a nonCFG node.
          delete mergedCFG;
          em.deallocateAllEdges();
          TCTNonCFGProfileNode* prof1 = TCTNonCFGProfileNode::newNonCFGProfileNode(node1);
          TCTNonCFGProfileNode* prof2 = TCTNonCFGProfileNode::newNonCFGProfileNode(node2);
          TCTNonCFGProfileNode* mergedProf = mergeProfileNode(prof1, weight1, prof2, weight2, ifAccumulate, isScoreOnly, isSum);
          delete prof1;
          delete prof2;
          return mergedProf;
        }
        
        if (compare == 0) // Process the child that proceeds in control flow.
          compare = node1->cfgGraph->compareChild(node1->getChild(k1)->ra, node2->getChild(k2)->ra);
        
        if (compare == 0) { // Tie breaker with TCTID.
          if (node1->getChild(k1)->id < node2->getChild(k2)->id) compare = -1;
          else compare = 1;
        }
        
        if (compare < 0) { // Process node1[k1] first
          const TCTANode* child1 = node1->getChild(k1);
          
          // Create dummy child2 node
          //TODO: set start/end time for the dummy node when isSum == false;
          TCTANode* child2 = child1->voidDuplicate();
          child2->setWeight(weight2);
          child2->initPerfLossMetric();
          
          TCTANode* mergedChild = mergeNode(child1, weight1, child2, weight2, ifAccumulate, isScoreOnly, isSum);
          inclusiveDiff += mergedChild->getDiffScore().getInclusive(); // Add child diff: 1.3) when not accumulating and 1) when accumulating

          if (isScoreOnly) delete mergedChild;
          else {
            mergedCFG->addChild(mergedChild);
            // Check all pending edges
            em.checkPendingEdges(node1->cfgGraph, mergedChild);
            // Process inEdges of the merged child
            em.completePendingEdge(mergedChild);
            // Process outEdges of the child1
            for (auto it = node1->getOutEdges(k1).begin(); it != node1->getOutEdges(k1).end(); it++)
              em.insertPendingEdge((*it)->duplicate(), mergedChild);
          }
          
          delete child2;
          
          k1++;
        }
        else { // Process node2[k2] first
          const TCTANode* child2 = node2->getChild(k2);
          
          // Create dummy child1 node
          //TODO: set start/end time for the dummy node when isSum == false;
          TCTANode* child1 = child2->voidDuplicate();
          child1->setWeight(weight1);
          child1->initPerfLossMetric();
          
          TCTANode* mergedChild = mergeNode(child1, weight1, child2, weight2, ifAccumulate, isScoreOnly, isSum);
          inclusiveDiff += mergedChild->getDiffScore().getInclusive(); // Add child diff: 1.3) when not accumulating and 1) when accumulating

          if (isScoreOnly) delete mergedChild;
          else {
            mergedCFG->addChild(mergedChild);
            // Check all pending edges
            em.checkPendingEdges(node1->cfgGraph, mergedChild);
            // Process inEdges of the merged child
            em.completePendingEdge(mergedChild);
            // Process outEdges of the child2
            for (auto it = node2->getOutEdges(k2).begin(); it != node2->getOutEdges(k2).end(); it++)
              em.insertPendingEdge((*it)->duplicate(), mergedChild);
          }
          delete child1;

          k2++;
        }
      }
    }
    
    // Add edges
    if (!isScoreOnly) {
      em.completePendingEdge(&TCTACFGNode::dummyEndNode);
      mergedCFG->setEdges(em.getAllEdges());
    }
    
    // Set to CFG profile if necessary
    if (node1->isCFGProfile() || node2->isCFGProfile())
      mergedCFG->toCFGProfile();
    
    if (!isSum) {
      // Compute diff among gaps
      Time minExclusive1, maxExclusive1, minExclusive2, maxExclusive2;
      node1->getExclusiveDuration(minExclusive1, maxExclusive1);
      node2->getExclusiveDuration(minExclusive2, maxExclusive2);
      inclusiveDiff += (double)computeRangeDiff(minExclusive1, maxExclusive1, minExclusive2, maxExclusive2, samplingPeriod)
              * (double) weight1 * (double) weight2;  // belongs to 2.3)

      /**
       * The exclusive difference score of a node is the sum of difference at this node (not counting is children),
       * which is the sum of difference of inclusive duration among all nodes represented by the merged CFG node.
       * When accumulating, exclusive difference score of the merged profile node consists of the following components -- 
       *  1) the difference of inclusive duration among all nodes represented by node1
       *  2) the difference of inclusive duration among all nodes represented by node2
       *  3) the difference of inclusive duration between nodes represented by node1 and node2
       * 
       * When not accumulating, only 3) is needed.
       */
      double exclusiveDiff = (double)computeRangeDiff(node1->getMinDuration(), node1->getMaxDuration(), 
                                                      node2->getMinDuration(), node2->getMaxDuration(), samplingPeriod)
              * (double) weight1 * (double) weight2; // Add 3)

      if (ifAccumulate) {
        // Add 1) and 2) to exclusive diff score.
        exclusiveDiff += node1->getDiffScore().getExclusive() + node2->getDiffScore().getExclusive();

        // The above code has added 1) and 2.3) to inclusive diff score. Add 2.1) and 2.2) to inclusive diff score below.
        // Add 2.1)
        inclusiveDiff += node1->getDiffScore().getInclusive();
        for (int i = 0; i < node1->getNumChild(); i++)
          inclusiveDiff -= node1->getChild(i)->getDiffScore().getInclusive();

        // Add 2.2)
        inclusiveDiff += node2->getDiffScore().getInclusive();
        for (int i = 0; i < node2->getNumChild(); i++)
          inclusiveDiff -= node2->getChild(i)->getDiffScore().getInclusive();
      }

      /* Inclusive diff score should be no less than "exclusive diff score" plus minimum accumulation.
       * (As we hide noises from sampling, the resulting inclusive diff score could be smaller than 
       * the minimum threshold)
       */
      double minInclusiveDiff = (double)computeRangeDiff(node1->getMinDuration(), node1->getMaxDuration(), 
                                                      node2->getMinDuration(), node2->getMaxDuration(), samplingPeriod)
              * (double) weight1 * (double) weight2;
      if (ifAccumulate)
        minInclusiveDiff += node1->getDiffScore().getInclusive() + node2->getDiffScore().getInclusive();

      if (inclusiveDiff < minInclusiveDiff) inclusiveDiff = minInclusiveDiff;

      mergedCFG->getDiffScore().setScores(inclusiveDiff, exclusiveDiff);
    }
    
    return mergedCFG;
  }
  /*
  TCTANode* DifferenceQuantifier::mergeTraceNode(const TCTATraceNode* trace1, long weight1, const TCTATraceNode* trace2, long weight2, 
          bool ifAccumulate, bool isScoreOnly) {
    TCTATraceNode* mergedTrace = (TCTATraceNode*)trace1->voidDuplicate();
    mergedTrace->getTime().setAsAverageTime(trace1->getTime(), weight1, trace2->getTime(), weight2);
    if (!isScoreOnly) {
      mergedTrace->setWeight(weight1 + weight2);
      mergedTrace->setRetCount(trace1->getRetCount() + trace2->getRetCount());
      mergedTrace->setDerivedSemanticLabel(trace1->getDerivedSemanticLabel() & trace2->getDerivedSemanticLabel());
      mergedTrace->getPerfLossMetric().setDuratonMetric(trace1->getPerfLossMetric(), trace2->getPerfLossMetric());
    }
    
    double inclusiveDiff = 0;
        
    Time samplingPeriod = mergedTrace->getSamplingPeriod();
    
    Time gapDiffMin = 0; // minimun value of gapDiff = (gap1 - gap2)
    Time gapDiffMax = 0; // maximun value of gapDiff = (gap1 - gap2)
    
    int k1 = 0, k2 = 0;
    
    Time startExclusive1 = trace1->getTime().getStartTimeExclusive();
    Time startInclusive1 = trace1->getTime().getStartTimeInclusive();
    Time startExclusive2 = trace2->getTime().getStartTimeExclusive();
    Time startInclusive2 = trace2->getTime().getStartTimeInclusive();
    
    while (k1 < trace1->getNumChild() || k2 < trace2->getNumChild()) {
      // When children of trace1 and trace2 match
      if (k1 < trace1->getNumChild() && k2 < trace2->getNumChild()
              && trace1->getChild(k1)->id == trace2->getChild(k2)->id) {
        // Add gap diff 2.3)
        Time minGap1, maxGap1, minGap2, maxGap2;
        trace1->getGapBeforeChild(k1, minGap1, maxGap1);
        trace2->getGapBeforeChild(k2, minGap2, maxGap2);
        inclusiveDiff += (double)computeRangeDiff(minGap1+gapDiffMin, maxGap1+gapDiffMax, minGap2, maxGap2, samplingPeriod)
            * (double) weight1 * (double) weight2;
        gapDiffMin = 0;
        gapDiffMax = 0;
        
        // Add child diff: 1.3) when not accumulating and 1) when accumulating
        const TCTANode* child1 = trace1->getChild(k1);
        const TCTANode* child2 = trace2->getChild(k2);
        TCTANode* mergedChild = mergeNode(child1, weight1, child2, weight2, ifAccumulate, isScoreOnly);
        inclusiveDiff += mergedChild->getDiffScore().getInclusive();
        
        if (isScoreOnly) delete mergedChild;
        else mergedTrace->addChild(mergedChild);
        
        k1++; k2++;
        trace1->getLastChildEndTime(k1, startExclusive1, startInclusive1);
        trace2->getLastChildEndTime(k2, startExclusive2, startInclusive2);
      }
      // When children of trace1 and trace2 don't match
      else {
        int compare = 0; // compare < 0 means trace1[k1] should be processed first; > 0 otherwise.
        if (k1 == trace1->getNumChild()) compare = 1;
        if (k2 == trace2->getNumChild()) compare = -1;
        
        if (compare == 0  
            && (trace1->cfgGraph == NULL // When cfgGraph is NULL
               || !trace1->cfgGraph->hasChild(trace1->getChild(k1)->ra) // or RA of trace1[k1] is not found in cfgGraph
               || !trace1->cfgGraph->hasChild(trace2->getChild(k2)->ra) // or RA of trace2[k2] is not found in cfgGraph
               )) { // Return a merged profile node.
          delete mergedTrace;
          TCTNonCFGProfileNode* prof1 = TCTNonCFGProfileNode::newNonCFGProfileNode(trace1);
          TCTNonCFGProfileNode* prof2 = TCTNonCFGProfileNode::newNonCFGProfileNode(trace2);
          TCTNonCFGProfileNode* mergedProf = mergeProfileNode(prof1, weight1, prof2, weight2, ifAccumulate, isScoreOnly);
          delete prof1;
          delete prof2;
          return mergedProf;
        }
        
        if (compare == 0) // Process the child that proceeds in control flow.
          compare = trace1->cfgGraph->compareChild(trace1->getChild(k1)->ra, trace2->getChild(k2)->ra);
        
        if (compare == 0) { // Tie breaker with TCTID.
          if (trace1->getChild(k1)->id < trace2->getChild(k2)->id) compare = -1;
          else compare = 1;
        }
        
        if (compare < 0) { // Process trace1[k1] first
          const TCTANode* child1 = trace1->getChild(k1);
          
          // Create dummy child2 node
          TCTANode* child2 = child1->voidDuplicate();
          child2->setWeight(weight2);
          // Set start/end time for the dummy child2 node.
          // Assumes that the gap between child2 and the latest matching child is the same as 
          // the gap between child1 and the latest matching child.
          Time startExclusive = startExclusive2 + (child1->getTime().getStartTimeExclusive() - startExclusive1);
          Time startInclusive = startInclusive2 + (child1->getTime().getStartTimeInclusive() - startInclusive1);
          
          Time startExclusiveMin, startInclusiveMin; // child2 should never start earlier than the last child of trace2.
          trace2->getLastChildEndTime(k2, startExclusiveMin, startInclusiveMin);
          startExclusive = max(startExclusive, startExclusiveMin);
          startInclusive = max(startInclusive, startInclusiveMin);
          
          Time startExclusiveMax, startInclusiveMax; // child2 should never start later than the current child of trace2.
          trace2->getCurrChildStartTime(k2, startExclusiveMax, startInclusiveMax);
          startExclusive = min(startExclusive, startExclusiveMax);
          startInclusive = min(startInclusive, startInclusiveMax);
          
          child2->getTime().setStartTime(startExclusive, startInclusive);
          child2->getTime().setEndTime(startExclusive, startInclusive);
          
          child2->initPerfLossMetric();
          
          TCTANode* mergedChild = mergeNode(child1, weight1, child2, weight2, ifAccumulate, isScoreOnly);
          inclusiveDiff += mergedChild->getDiffScore().getInclusive(); // Add child diff: 1.3) when not accumulating and 1) when accumulating

          if (isScoreOnly) delete mergedChild;
          else mergedTrace->addChild(mergedChild);
          
          delete child2;
          
          // Accumulate gap for next match
          Time gapMin, gapMax;
          trace1->getGapBeforeChild(k1, gapMin, gapMax);
          gapDiffMin += gapMin;
          gapDiffMax += gapMax;
          
          k1++;
        }
        else { // Process trace2[k2] first
          const TCTANode* child2 = trace2->getChild(k2);
          
          // Create dummy child1 node
          TCTANode* child1 = child2->voidDuplicate();
          child1->setWeight(weight1);
          // Set start/end time for the dummy child1 node.
          // Assumes that the gap between child1 and the latest matching child is the same as 
          // the gap between child2 and the latest matching child.
          Time startExclusive = startExclusive1 + (child2->getTime().getStartTimeExclusive() - startExclusive2);
          Time startInclusive = startInclusive1 + (child2->getTime().getStartTimeInclusive() - startInclusive2);
          
          Time startExclusiveMin, startInclusiveMin; // child1 should never start earlier than the last child of trace1.
          trace1->getLastChildEndTime(k1, startExclusiveMin, startInclusiveMin);
          startExclusive = max(startExclusive, startExclusiveMin);
          startInclusive = max(startInclusive, startInclusiveMin);
                    
          Time startExclusiveMax, startInclusiveMax; // child1 should never start later than the current child of trace1.
          trace1->getCurrChildStartTime(k1, startExclusiveMax, startInclusiveMax);
          startExclusive = min(startExclusive, startExclusiveMax);
          startInclusive = min(startInclusive, startInclusiveMax);
          
          // Set the midpoints as start/end time for the dummy node.
          child1->getTime().setStartTime(startExclusive, startInclusive);
          child1->getTime().setEndTime(startExclusive, startInclusive);
          
          child1->initPerfLossMetric();
          
          TCTANode* mergedChild = mergeNode(child1, weight1, child2, weight2, ifAccumulate, isScoreOnly);
          inclusiveDiff += mergedChild->getDiffScore().getInclusive(); // Add child diff: 1.3) when not accumulating and 1) when accumulating

          if (isScoreOnly) delete mergedChild;
          else mergedTrace->addChild(mergedChild);
          
          delete child1;
          
          // Accumulate gap for next match
          Time gapMin, gapMax;
          trace2->getGapBeforeChild(k2, gapMin, gapMax);
          gapDiffMin -= gapMax;
          gapDiffMax -= gapMin;
          
          k2++;
        }
      }
    }
    
    // Final gap
    Time gapMin, gapMax;
    trace1->getGapBeforeChild(k1, gapMin, gapMax);
    gapDiffMin += gapMin;
    gapDiffMax += gapMax;
    
    trace2->getGapBeforeChild(k2, gapMin, gapMax);
    gapDiffMin -= gapMax;
    gapDiffMax -= gapMin;
    
    inclusiveDiff += (double)computeRangeDiff(gapDiffMin, gapDiffMax, 0, 0, samplingPeriod)
            * (double) weight1 * (double) weight2; // Add gap diff 2.3)
    
    double exclusiveDiff = (double)computeRangeDiff(trace1->getMinDuration(), trace1->getMaxDuration(), 
                                                    trace2->getMinDuration(), trace2->getMaxDuration(), samplingPeriod)
            * (double) weight1 * (double) weight2; // Add 3)
    
    if (ifAccumulate) {
      // Add 1) and 2) to exclusive diff score.
      exclusiveDiff += trace1->getDiffScore().getExclusive() + trace2->getDiffScore().getExclusive();
      
      // The above code has added 1) and 2.3) to inclusive diff score. Add 2.1) and 2.2) to inclusive diff score below.
      // Add 2.1)
      inclusiveDiff += trace1->getDiffScore().getInclusive();
      for (int i = 0; i < trace1->getNumChild(); i++)
        inclusiveDiff -= trace1->getChild(i)->getDiffScore().getInclusive();
      
      // Add 2.2)
      inclusiveDiff += trace2->getDiffScore().getInclusive();
      for (int i = 0; i < trace2->getNumChild(); i++)
        inclusiveDiff -= trace2->getChild(i)->getDiffScore().getInclusive();
    }

    double minInclusiveDiff = (double)computeRangeDiff(trace1->getMinDuration(), trace1->getMaxDuration(), 
                                                    trace2->getMinDuration(), trace2->getMaxDuration(), samplingPeriod)
            * (double) weight1 * (double) weight2;
    if (ifAccumulate)
      minInclusiveDiff += trace1->getDiffScore().getInclusive() + trace2->getDiffScore().getInclusive();
    
    if (inclusiveDiff < minInclusiveDiff) inclusiveDiff = minInclusiveDiff;
    
    mergedTrace->getDiffScore().setScores(inclusiveDiff, exclusiveDiff);
    
    return mergedTrace;
  }
  */
  TCTANode* DifferenceQuantifier::mergeLoopNode(const TCTLoopNode* loop1, long weight1, const TCTLoopNode* loop2, long weight2, 
          bool ifAccumulate, bool isScoreOnly, bool isSum) {
    // If either loop is not accepted 
    if ((!loop1->accept()) || (!loop2->accept()) 
            || (loop1->getNumIteration() != loop2->getNumIteration())
            || (loop1->getNumAcceptedIteration() != loop2->getNumAcceptedIteration())
            || isSum)
      return mergeNode(loop1->getProfileNode(), weight1, loop2->getProfileNode(), weight2, ifAccumulate, isScoreOnly, isSum);
    
    TCTLoopNode* mergedLoop = NULL;
    TCTANode* mergedProf = NULL;
    if (!isScoreOnly) {
      mergedLoop = new TCTLoopNode(*loop1, weight1, *loop2, weight2, ifAccumulate);
      mergedProf = mergedLoop->getProfileNode();
    } 
    else {
      mergedProf = mergeNode(loop1->getProfileNode(), weight1, loop2->getProfileNode(), weight2, ifAccumulate, isScoreOnly, isSum);
    }
    TCTANode* minCopy = mergedProf->duplicate();
    mergedProf->clearDiffScore();
    
    const TCTClusterNode* cluster1 = loop1->getClusterNode();
    const TCTClusterNode* cluster2 = loop2->getClusterNode();
    
    int numCluster1 = cluster1->getNumClusters();
    int numCluster2 = cluster2->getNumClusters();
    
    const TCTANode* rep1[MAX_NUM_CLUSTER];
    const TCTANode* rep2[MAX_NUM_CLUSTER];
    long numMembers1[MAX_NUM_CLUSTER];
    long numMembers2[MAX_NUM_CLUSTER];
    
    for (int i = 0; i < numCluster1; i++) {
      rep1[i] = cluster1->getClusterRepAt(i);
      numMembers1[i] = rep1[i]->getWeight() * loop2->getWeight();
    }
    for (int i = 0; i < numCluster2; i++) {
      rep2[i] = cluster2->getClusterRepAt(i);
      numMembers2[i] = rep2[i]->getWeight() * loop1->getWeight();
    }
    
    TCTANode* diffNodes[MAX_NUM_CLUSTER][MAX_NUM_CLUSTER];
    double diffRatio[MAX_NUM_CLUSTER][MAX_NUM_CLUSTER];
    
    for (int i = 0; i < numCluster1; i++)
      for (int j = 0; j < numCluster2; j++) {
        diffNodes[i][j] = mergeNode(rep1[i], 1, rep2[j], 1, false, isScoreOnly, isSum);
        diffRatio[i][j] = diffNodes[i][j]->getDiffScore().getInclusive() / (double)(rep1[i]->getDuration() + rep2[j]->getDuration());
      }
    
    while (numCluster1 > 0 && numCluster2 > 0) {
      int idx1 = 0, idx2 = 0;
      for (int i = 0; i < numCluster1; i++)
        for (int j = 0; j < numCluster2; j++)
          if (diffRatio[i][j] < diffRatio[idx1][idx2]) {
            idx1 = i; idx2 = j;
          }
      
      long numMatched = min(numMembers1[idx1], numMembers2[idx2]);
      addDiffScore(mergedProf, diffNodes[idx1][idx2], 
              (double)numMatched * (double)weight1 * (double) weight2 / (double)loop1->getWeight() / (double)loop2->getWeight());
      
      numMembers1[idx1] -= numMatched;
      numMembers2[idx2] -= numMatched;
      
      if (numMembers1[idx1] == 0) {
        numCluster1--;
        rep1[idx1] = rep1[numCluster1];
        numMembers1[idx1] = numMembers1[numCluster1];
        for (int j = 0; j < numCluster2; j++) {
          delete diffNodes[idx1][j];
          diffNodes[idx1][j] = diffNodes[numCluster1][j];
          diffRatio[idx1][j] = diffRatio[numCluster1][j];
        }
      }
      
      if (numMembers2[idx2] == 0) {
        numCluster2--;
        rep2[idx2] = rep2[numCluster2];
        numMembers2[idx2] = numMembers2[numCluster2];
        for (int i = 0; i < numCluster1; i++) {
          delete diffNodes[i][idx2];
          diffNodes[i][idx2] = diffNodes[i][numCluster2];
          diffRatio[i][idx2] = diffRatio[i][numCluster2];
        }
      }
    }
    
    while (numCluster1 > 0) {
      numCluster1--;
      TCTANode* dummyNode = rep1[numCluster1]->voidDuplicate();
      TCTANode* diffNode = mergeNode(rep1[numCluster1], 1, dummyNode, 1, false, isScoreOnly, isSum);
      addDiffScore(mergedProf, diffNode, 
              (double)numMembers1[numCluster1] * (double)weight1 * (double) weight2 / (double)loop1->getWeight() / (double)loop2->getWeight());
      delete dummyNode;
      delete diffNode;
    }
    
    while (numCluster2 > 0) {
      numCluster2--;
      TCTANode* dummyNode = rep2[numCluster2]->voidDuplicate();
      TCTANode* diffNode = mergeNode(rep2[numCluster2], 1, dummyNode, 1, false, isScoreOnly, isSum);
      addDiffScore(mergedProf, diffNode, 
              (double)numMembers2[numCluster2] * (double)weight1 * (double) weight2 / (double)loop1->getWeight() / (double)loop2->getWeight());
      delete dummyNode;
      delete diffNode;
    }
    
    const TCTANode* rej1 = loop1->getRejectedIterations();
    const TCTANode* rej2 = loop2->getRejectedIterations();
    if (rej1 != NULL || rej2 != NULL) {
      if (rej1 == NULL) rej1 = rej2->voidDuplicate();
      if (rej2 == NULL) rej2 = rej1->voidDuplicate();
      TCTANode* rejDiff = diffQ.mergeNode(rej1, weight1, rej2, weight2, false, isScoreOnly, isSum);
      addDiffScore(mergedProf, rejDiff, 1);
      delete rejDiff;
      if (loop1->getRejectedIterations() == NULL) delete rej1;
      if (loop2->getRejectedIterations() == NULL) delete rej2;
    }
    
    if (ifAccumulate) {
      addDiffScore(mergedProf, loop1->getProfileNode(), 1);
      addDiffScore(mergedProf, loop2->getProfileNode(), 1);
    }
    
    if (mergedProf->getDiffScore().getInclusive() < minCopy->getDiffScore().getInclusive()) {
      mergedProf->clearDiffScore();
      addDiffScore(mergedProf, minCopy, 1);
    }
    delete minCopy;

    if (mergedLoop != NULL) {
      mergedLoop->getDiffScore().setScores(mergedProf->getDiffScore().getInclusive(), mergedProf->getDiffScore().getExclusive());
      return mergedLoop;
    }
    else 
      return mergedProf;
  }
  
  void DifferenceQuantifier::addDiffScore(TCTANode* dst, const TCTANode* src, double ratio) {
    if (src->getDiffScore().getInclusive() == 0) return;
    
    if (dst->type == TCTANode::NonCFGProf) {
      if (src->type == TCTANode::NonCFGProf) 
        addDiffScore((TCTNonCFGProfileNode*)dst, (TCTNonCFGProfileNode*)src, ratio);
      else {
        TCTNonCFGProfileNode* temp = TCTNonCFGProfileNode::newNonCFGProfileNode(src);
        addDiffScore((TCTNonCFGProfileNode*)dst, temp, ratio);
        delete temp;
      }
    } 
    else if (dst->type == TCTANode::Loop) 
      addDiffScore((TCTLoopNode*)dst, (TCTLoopNode*)src, ratio);
    else if (src->type == TCTANode::Loop)
      addDiffScore((TCTACFGNode*)dst, ((TCTLoopNode*)src)->getProfileNode(), ratio);
    else
      addDiffScore((TCTACFGNode*)dst, (TCTACFGNode*)src, ratio);
  }
  
  void DifferenceQuantifier::addDiffScore(TCTACFGNode* dst, const TCTACFGNode* src, double ratio) {
    double inclusive = dst->getDiffScore().getInclusive() + src->getDiffScore().getInclusive() * ratio;
    double exclusive = dst->getDiffScore().getExclusive() + src->getDiffScore().getExclusive() * ratio;
    dst->getDiffScore().setScores(inclusive, exclusive);
    
    int k1 = 0, k2 = 0;
    while (k1 < dst->getNumChild() && k2 < src->getNumChild())
      if (dst->getChild(k1)->id == src->getChild(k2)->id) {
        addDiffScore(dst->getChild(k1), src->getChild(k2), ratio);
        k1++; k2++;
      }
      else
        k1++;
    
    if (k2 != src->getNumChild())
      print_msg(MSG_PRIO_NORMAL, "ERROR: failed to add diff scores of all children for trace node %s.\n", src->id.toString().c_str());
  }
  /*
  void DifferenceQuantifier::addDiffScore(TCTATraceNode* dst, const TCTATraceNode* src, double ratio) {
    double inclusive = dst->getDiffScore().getInclusive() + src->getDiffScore().getInclusive() * ratio;
    double exclusive = dst->getDiffScore().getExclusive() + src->getDiffScore().getExclusive() * ratio;
    dst->getDiffScore().setScores(inclusive, exclusive);
    
    int k1 = 0, k2 = 0;
    while (k1 < dst->getNumChild() && k2 < src->getNumChild())
      if (dst->getChild(k1)->id == src->getChild(k2)->id) {
        addDiffScore(dst->getChild(k1), src->getChild(k2), ratio);
        k1++; k2++;
      }
      else
        k1++;
    
    if (k2 != src->getNumChild())
      print_msg(MSG_PRIO_NORMAL, "ERROR: failed to add diff scores of all children for trace node %s.\n", src->id.toString().c_str());
  }*/
  
  void DifferenceQuantifier::addDiffScore(TCTNonCFGProfileNode* dst, const TCTNonCFGProfileNode* src, double ratio) {
    double inclusive = dst->getDiffScore().getInclusive() + src->getDiffScore().getInclusive() * ratio;
    double exclusive = dst->getDiffScore().getExclusive() + src->getDiffScore().getExclusive() * ratio;
    dst->getDiffScore().setScores(inclusive, exclusive);
    
    for (auto it = src->getChildMap().begin(); it != src->getChildMap().end(); it++)
      if (dst->getChildMap().find(it->first) != dst->getChildMap().end())
        addDiffScore(dst->getChildMap().at(it->first), it->second, ratio);
      else
        print_msg(MSG_PRIO_NORMAL, "ERROR: failed to add diff scores of children %s for profile node %s.\n", 
                it->first.toString().c_str(), src->id.toString().c_str());
  }
  
  void DifferenceQuantifier::addDiffScore(TCTLoopNode* dst, const TCTLoopNode* src, double ratio) {
    double inclusive = dst->getDiffScore().getInclusive() + src->getDiffScore().getInclusive() * ratio;
    double exclusive = dst->getDiffScore().getExclusive() + src->getDiffScore().getExclusive() * ratio;
    dst->getDiffScore().setScores(inclusive, exclusive);
    
    addDiffScore(dst->getProfileNode(), src->getProfileNode(), ratio);
  }
  
  TCTANode* DifferenceQuantifier::computeAvgRep(const TCTClusterNode* cluster) {
    if (cluster->getNumClusters() == 0) return NULL;
    
    TCTANode* avgRep = cluster->getClusterRepAt(0)->duplicate();
    for (int k = 1; k < cluster->getNumClusters(); k++) {
      TCTANode* temp = mergeNode(avgRep, avgRep->getWeight(), cluster->getClusterRepAt(k), cluster->getClusterRepAt(k)->getWeight(), false, false, false);
      delete avgRep;
      avgRep = temp;
    }
    avgRep->clearDiffScore();
    
    for (int k = 0; k < cluster->getNumClusters(); k++)
      addDiffScore(avgRep, cluster->getClusterRepAt(k), 1);
    
    for (int i = 0; i < cluster->getNumClusters(); i++)
      for (int j = i+1; j < cluster->getNumClusters(); j++) {
        TCTANode* temp = mergeNode(cluster->getClusterRepAt(i), cluster->getClusterRepAt(i)->getWeight(), cluster->getClusterRepAt(j), cluster->getClusterRepAt(j)->getWeight(), false, false, false);
        addDiffScore(avgRep, temp, 1);
        delete temp;
      }
    
    return avgRep;
  }
  
  DifferenceQuantifier diffQ;
}